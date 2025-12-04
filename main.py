import asyncio
import os
from datetime import datetime
from typing import Optional, Type, Literal
from zoneinfo import ZoneInfo

from google.adk.agents import Agent, LoopAgent, SequentialAgent, ParallelAgent
from google.adk.events import Event
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search, FunctionTool, ToolContext
from google.genai import types
from pydantic import BaseModel, Field
import yfinance as yf
import pandas as pd
import pandas_ta as ta

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

RETRY_CONFIG = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,  # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)
DEFAULT_AGENT_DESCRIPTION = "A simple agent that can answer general questions."
DEFAULT_AGENT_INSTRUCTION = (
    "You are a helpful assistant. Use Google Search for current info or if unsure."
)


class Stock(BaseModel):
    name: str
    ticker: str
    summary: str
    current_price: float
    upside_percent: float

class StocksPick(BaseModel):
    stocks: list[Stock]

class StocksReport(BaseModel):
    date: str
    stocks: StocksPick

class CriticResponse(BaseModel):
    status: Literal["APPROVED", "REJECTED"]
    suggestions: list[str] = Field(default_factory=list)
    reasoning: str


def get_intraday_technicals(ticker: str) -> str:
    """
    Fetches 5-minute INTRADAY price data (including Pre-Market).
    Calculates VWAP, RSI, and Momentum for Day Trading.
    """
    try:
        # 1. Fetch Data
        df = yf.download(
            tickers=ticker,
            period="5d",
            interval="5m",
            prepost=True,
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            return f"No intraday data found for {ticker}."

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 2. Calculate Indicators (Explicit assignment)
        if len(df) > 0:
            # VWAP
            vwap_series = ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"])
            df["VWAP_D"] = vwap_series

            # RSI
            df["RSI_14"] = ta.rsi(df["Close"], length=14)

            # ATR
            df["ATRr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

            # EMA
            df["EMA_9"] = ta.ema(df["Close"], length=9)

        # 3. Get Latest Candle
        latest = df.iloc[-1]

        # 4. Extract values safely with defaults
        # Using (val or 0.0) handles None/NaN safely
        price = float(latest["Close"])
        vwap_val = float(latest.get("VWAP_D") or 0.0)
        ema_9 = float(latest.get("EMA_9") or 0.0)
        rsi = float(latest.get("RSI_14") or 0.0)
        atr = float(latest.get("ATRr_14") or 0.0)

        # Determine Trend
        if vwap_val == 0:
            trend = "NEUTRAL (No VWAP)"
        else:
            trend = (
                "BULLISH (Above VWAP)" if price > vwap_val else "BEARISH (Below VWAP)"
            )

        summary = f"""
        INTRADAY SNAPSHOT for {ticker.upper()}
        Price: ${price:.2f} | Time: {latest.name.strftime('%H:%M EST')}
    
        [Trend Status]
        - Signal: {trend}
        - VWAP (Institutional Support): ${vwap_val:.2f}
        - 9 EMA (Fast Trend): ${latest['EMA_9']:.2f}
    
        [Momentum]
        - RSI (14): {latest['RSI_14']:.2f} (Over 70=Hot, Under 30=Cold)
        - Volatility (ATR): ${latest['ATRr_14']:.2f} per candle
    
        [Volume]
        - Current Candle Vol: {latest['Volume']:,}
        """

        return summary
    except Exception as e:
        # Print full trace for debugging if it fails again
        import traceback
        traceback.print_exc()
        return f"Error fetching data for {ticker}: {e}"


def generate_agent(
    name: str,
    model: str = "gemini-2.5-flash-lite",
    description: str = DEFAULT_AGENT_DESCRIPTION,
    instruction: str = DEFAULT_AGENT_INSTRUCTION,
    tools: list = [],
    output_key: str = None,
    output_schema: Optional[Type[BaseModel]] = None,
    generation_config: Optional[types.GenerateContentConfig] = None,
) -> Agent:
    if generation_config is None:
        generation_config = types.GenerateContentConfig(temperature=0.0)
    agent = Agent(
        name=name,
        model=Gemini(model=model, retry_options=RETRY_CONFIG),
        description=description,
        instruction=instruction,
        tools=tools,
        output_key=output_key,
        generate_content_config=generation_config,
    )
    if output_schema is not None:
        agent.output_schema = output_schema
    return agent


def generate_runner(
    agent: Agent | SequentialAgent | LoopAgent | ParallelAgent,
) -> InMemoryRunner:
    return InMemoryRunner(agent=agent)


def get_last_event_message(final_state: list[Event]) -> str:
    message = ""
    for part in final_state[-1].content.parts:
        message = message + part.text
    return message


def exit_loop(tool_context: ToolContext) -> str:
    tool_context.actions.escalate = True
    return "APPROVED"


async def debug_stocks_run(date: str) -> None:
    stocks_search_agent = generate_agent(
        name="StocksSearchAgent",
        description="An agent that fetches stocks from google search.",
        instruction=f"""
        CONTEXT: It is currently PRE-MARKET on {date}. 
        GOAL: Identify stocks to BUY at the Open (9:30 AM) and SELL at the Close (4:00 PM) today.
        
        Your Strategy:
        1. Search for "Top Pre-Market Gainers {date}" and "Stocks gapping up today".
        2. Look for specific catalysts released LAST NIGHT or THIS MORNING:
           - Earnings Beats (reported pre-market or after-hours yesterday).
           - FDA Approvals / Biotech News.
           - New Contracts or Mergers.
        3. IGNORE long-term investments. IGNORE crypto.
        
        Return a text summary of 3-5 stocks that have HIGH PRE-MARKET VOLUME and a clear catalyst.
        """,
        tools=[google_search],
        output_key="stocks_findings",
    )
    stocks_critic_agent = generate_agent(
        name="StocksCriticAgent",
        instruction="""
        Review the data in: {stocks_findings}
        
        STRICT DAY-TRADING CRITERIA:
        - REJECT any stock that does not have "Breaking News" from the last 12 hours.
        - REJECT stocks with low relative volume (RVOL). We need liquidity to enter/exit.
        - REJECT "Blue Chips" (like AAPL/MSFT) unless they have a massive specific catalyst today.
        
        Use Google Search to verify the catalyst time and relevance.
        - If the stock suggestions are correct cased on the criteria, you MUST return the exact phrase: "APPROVED"
        - Otherwise, list with 2-3 specific, actionable suggestions for improvement.
        """,
        tools=[google_search],
        output_key="stocks_critic",
    )
    stocks_refiner_agent = generate_agent(
        name="StocksRefinerAgent",
        description="An agent that decides whether the chosen stocks are actually the hottest stocks for today.",
        instruction="""
        You are a stock choosing refiner, you have
        Hottest Stocks: {stocks_findings}
        Critic: {stocks_critic}
        Your task is to analyze the critic.
        - IF the critic is EXACTLY "APPROVED", you MUST call the `exit_loop` function and nothing else.
        - OTHERWISE, find better pre-market gappers using the suggestions.
        """,
        tools=[FunctionTool(exit_loop)],
        output_key="stocks_findings",
    )
    stock_choosing_refinement_loop = LoopAgent(
        name="StocksRefinementLoop",
        sub_agents=[stocks_critic_agent, stocks_refiner_agent],
        max_iterations=5,  # Prevents infinite loops
    )
    stock_upside_evaluator_agent = generate_agent(
        name="StockEvaluatorAgent",
        description="An agent that takes the summerized stock results and adds daily upside projection.",
        instruction="""
        Read the stocks in: {stocks_findings}
        
        GOAL: Calculate Intraday Technical Upside using the math tool.
        
        1. For each ticker, CALL `get_intraday_technicals`.
        2. Analyze the returned data:
           - BULLISH: Price > VWAP.
           - BEARISH: Price < VWAP.
           - RISK: RSI > 70 (Overbought).
        
        3. Output a structured TEXT analysis for each stock:
           "Ticker: [Symbol] | Price: [Price]$ | Signal: [Bull/Bear] | RSI: [Val] | Upside: [Est %]"
           
        (If Bearish, Upside is 0%).
        """,
        tools=[FunctionTool(get_intraday_technicals)],
        output_key="stocks_evaluation",
    )
    evaluation_critic_agent = generate_agent(
        name="EvaluationCriticAgent",
        instruction="""
        Review Technicals in: {stocks_evaluation}
        
        Verify the math using `get_intraday_technicals`:
        1. Did they say "Bullish" when Price is actually BELOW VWAP?
        2. Did they ignore high risk (RSI > 75)?
        
        - If safe and accurate, return exactly: "APPROVED"
        - Otherwise, provide specific corrections text.
        
        - If the upside evaluation for a daily trade are somewhat safe, correct and complete, you MUST return the exact phrase: "APPROVED"
        - Otherwise, provide specific corrections text or list with 2-3 specific, actionable suggestions for improvement.""",
        tools=[FunctionTool(get_intraday_technicals)],
        output_key="evaluation_critic",
    )
    evaluation_refiner_agent = generate_agent(
        name="EvaluationRefinerAgent",
        description="An agent that decides whether the stocks upside evaluation are correct for today.",
        instruction="""
        You are a stock evaluation refiner, you have
        Hottest Stocks: {stocks_evaluation}
        Critic: {evaluation_critic}
        Your task is to analyze the critic.
        - IF the critic is EXACTLY "APPROVED", you MUST call the `exit_loop` function and nothing else.
        - OTHERWISE, adjust the upside % to to fully incorporate the feedback from the critic.
        """,
        tools=[FunctionTool(exit_loop)],
        output_key="stocks_evaluation",
    )
    stock_evaluation_refinement_loop = LoopAgent(
        name="StocksRefinementLoop",
        sub_agents=[evaluation_critic_agent, evaluation_refiner_agent],
        max_iterations=5,  # Prevents infinite loops
    )
    summarize_agent = generate_agent(
        name="SummarizerAgent",
        description="An agent that takes the chosen stock results and summarizes them.",
        instruction=f"""
        Data: {{stocks_evaluation}}
        
        Create a "Pre-Market Day Trading Plan".
        1. List the stocks.
        2. Extract the **Current Price** from the technical snapshot.
        3. Explicitly state the "Catalyst" and "Technicals".
        4. Date: {date}.
        """,
        tools=[],
        output_key="summary",
        output_schema=StocksReport
    )
    stock_root_agent = SequentialAgent(
        name="StocksChooserPipeline",
        sub_agents=[
            stocks_search_agent,
            stock_choosing_refinement_loop,
            stock_upside_evaluator_agent,
            stock_evaluation_refinement_loop,
            summarize_agent,
        ],
    )
    stock_root_runner = generate_runner(stock_root_agent)
    final_state = await stock_root_runner.run_debug("")
    print("Ended stock analysis")


if __name__ == "__main__":
    event_loop = asyncio.get_event_loop()
    ny_date = datetime.now(ZoneInfo("America/New_York")).date()
    event_loop.run_until_complete(
        debug_stocks_run(ny_date.strftime("%Y-%m-%d"))
    )

import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

from google.adk.agents import LoopAgent, SequentialAgent
from google.adk.tools import google_search, FunctionTool

from models import StocksReport
from utils import generate_agent, generate_runner, get_intraday_technicals, exit_loop


async def run_pipeline(date: str) -> None:
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
        output_schema=StocksReport,
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
    event_loop.run_until_complete(run_pipeline(ny_date.strftime("%Y-%m-%d")))

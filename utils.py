from typing import Optional, Type

from google.adk.agents import Agent, LoopAgent, SequentialAgent, ParallelAgent
from google.adk.tools import ToolContext
from google.adk.events import Event
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from pydantic import BaseModel

from config import RETRY_CONFIG, DEFAULT_AGENT_INSTRUCTION, DEFAULT_AGENT_DESCRIPTION


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

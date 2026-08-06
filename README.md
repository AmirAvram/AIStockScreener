# AIStockScreener

AIStockScreener is a Python stock-screening pipeline built with Google ADK/Gemini. It searches for pre-market stock movers, critiques and refines candidates, fetches intraday technical indicators with `yfinance` and `pandas-ta`, then prints a structured day-trading report.

This project is for research and automation experiments only. It is not financial advice.

## What it does

- Searches for pre-market gainers and stocks gapping up today.
- Filters for recent catalysts such as earnings, FDA/biotech news, contracts, mergers, or other breaking news.
- Uses a critic/refiner loop to reject weak or stale candidates.
- Pulls 5-minute intraday data with pre/post-market candles.
- Calculates VWAP, RSI, ATR, and EMA-based technical snapshots.
- Produces a final structured stock report.

## Requirements

- Python 3.12
- A Google API key available as `GOOGLE_API_KEY`
- Internet access for Google Search/Gemini calls and Yahoo Finance data
- A writable Numba cache directory when using `pandas-ta`

The checked-in `requirements.txt` is a pinned UTF-8 freeze from the working project virtual environment. The main direct runtime packages are:

- `google-adk`
- `google-genai`
- `pydantic`
- `pandas`
- `pandas-ta`
- `yfinance`

## Setup

From WSL/Linux:

```bash
cd /mnt/e/projects/AIStockScreener
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-compile -r requirements.txt
```

If `python3 -m venv .venv` fails with an `ensurepip` error on Ubuntu/Debian, install the venv package first:

```bash
sudo apt-get install python3.12-venv
```

In this workspace, the `.venv` was created from WSL. If you want to run the project with native Windows Python instead, recreate the venv from Windows because WSL virtual environments are not portable to PowerShell/CMD.

## Configuration

Set `GOOGLE_API_KEY` before running the app.

WSL/Linux:

```bash
export GOOGLE_API_KEY="your-google-api-key"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/aistockscreener-numba-cache"
```

PowerShell:

```powershell
$env:GOOGLE_API_KEY = "your-google-api-key"
$env:NUMBA_CACHE_DIR = "$env:TEMP\aistockscreener-numba-cache"
```

## Run

From WSL/Linux:

```bash
cd /mnt/e/projects/AIStockScreener
source .venv/bin/activate
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/aistockscreener-numba-cache"
python main.py
```

The app automatically uses the current date in the `America/New_York` timezone.

## Notes

- The pipeline calls live external services, so results can vary by market time, news availability, and API/data-provider responses.
- The code reads `GOOGLE_API_KEY` from the process environment in `config.py`.
- `NUMBA_CACHE_DIR` avoids import-time cache issues from `pandas-ta`/Numba when the venv lives on a mounted Windows drive or runs inside a restricted environment.
- Installing with `--no-compile` is recommended when the venv lives on a mounted Windows drive because it avoids slow bytecode compilation across the WSL mount.

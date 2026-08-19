# Stock Analytics Dashboard

A Dash + Plotly dashboard for flexible stock/ETF/index historical analysis using Yahoo Finance.

## Features

- Search/select an Indian stock, ETF, or index
- Custom Yahoo Finance ticker
- Automatically discovers the ticker's available historical range
- User-selected start/end dates
- Rolling-window analysis
- Closing/adjusted-close analysis
- Return, volatility, drawdown and descriptive statistics
- User-selected benchmark
- Historical repeated-period mode with one-sided t-test and Wilcoxon test
- Interactive Plotly charts and data table
- Render deployment configuration included

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.app
```

Then open http://127.0.0.1:8050

## Render

Create a new Web Service from this repository. Render can use `render.yaml`, or manually use:

Build command:
`pip install -r requirements.txt`

Start command:
`gunicorn app:server`

Yahoo Finance data is downloaded at runtime; no API key is required by this first version.

## V2 additions

- Compare up to 10 stocks/ETFs/indices simultaneously
- Automatic ticker mapping from company/security names
- Nifty 50 BeES (`NIFTYBEES.NS`) included
- Relative-performance chart for multi-stock comparisons
- Single-stock chart options: line, area, candlestick, OHLC
- Statistical analysis is restricted to one selected asset
- User-selected benchmark

## V3 chart behavior

- Raw price is the primary chart series.
- Rolling mean is an optional overlay on the raw price chart.
- The rolling mean is calculated from the selected Close/Adjusted Close price series.
- Returns are calculated separately from baseline/comparison values.
- Volume can be enabled as an optional chart series.

## V4 date behavior

- Uses a single DatePickerRange for easier start/end selection.
- Quick ranges: 1M, 3M, 6M, 1Y, 5Y and Max.
- The latest permitted date is based on the most recent Yahoo Finance data and today's date; it is not hard-coded to 19-Aug-2026.
- Changing stocks preserves the user's selected dates whenever those dates remain valid.
- When a newly selected stock has a shorter history, dates are automatically clamped to the common available range.

## V5 controls

- Dedicated Reset button.
- Reset restores GoldBeES, single-stock mode, 3-day rolling window, Close price, 1% benchmark, line chart, rolling-mean overlay, and an approximately one-year date range.
- Analyse is the primary action.
- Analysis output uses a loading indicator while processing.

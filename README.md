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
# Stocks-Researcher

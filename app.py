
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import ttest_1samp, wilcoxon
from dash import Dash, Input, Output, State, dcc, html, dash_table
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Stock universe
# ---------------------------------------------------------------------------

TICKERS = {
    "Indices": {
        "NIFTY 50": "^NSEI",
        "NIFTY BANK": "^NSEBANK",
        "NIFTY IT": "^CNXIT",
        "NIFTY AUTO": "^CNXAUTO",
        "NIFTY PHARMA": "^CNXPHARMA",
        "NIFTY FMCG": "^CNXFMCG",
        "NIFTY METAL": "^CNXMETAL",
    },
    "Banking & Financial": {
        "HDFC Bank": "HDFCBANK.NS",
        "ICICI Bank": "ICICIBANK.NS",
        "State Bank of India": "SBIN.NS",
        "Axis Bank": "AXISBANK.NS",
        "Kotak Mahindra Bank": "KOTAKBANK.NS",
        "IndusInd Bank": "INDUSINDBK.NS",
        "Bank of Baroda": "BANKBARODA.NS",
        "Punjab National Bank": "PNB.NS",
        "Canara Bank": "CANBK.NS",
        "Union Bank of India": "UNIONBANK.NS",
        "Bajaj Finance": "BAJFINANCE.NS",
        "Bajaj Finserv": "BAJAJFINSV.NS",
        "Shriram Finance": "SHRIRAMFIN.NS",
        "SBI Life": "SBILIFE.NS",
        "HDFC Life": "HDFCLIFE.NS",
    },
    "IT": {
        "TCS": "TCS.NS",
        "Infosys": "INFY.NS",
        "HCL Technologies": "HCLTECH.NS",
        "Wipro": "WIPRO.NS",
        "Tech Mahindra": "TECHM.NS",
        "LTIMindtree": "LTIM.NS",
        "Persistent Systems": "PERSISTENT.NS",
        "Mphasis": "MPHASIS.NS",
    },
    "Energy & Oil": {
        "Reliance Industries": "RELIANCE.NS",
        "ONGC": "ONGC.NS",
        "Indian Oil": "IOC.NS",
        "BPCL": "BPCL.NS",
        "Hindustan Petroleum": "HINDPETRO.NS",
        "GAIL": "GAIL.NS",
        "Adani Green Energy": "ADANIGREEN.NS",
        "NTPC": "NTPC.NS",
        "Power Grid": "POWERGRID.NS",
    },
    "Automobile": {
        "Maruti Suzuki": "MARUTI.NS",
        "Tata Motors": "TATAMOTORS.NS",
        "Mahindra & Mahindra": "M&M.NS",
        "Bajaj Auto": "BAJAJ-AUTO.NS",
        "Hero MotoCorp": "HEROMOTOCO.NS",
        "Eicher Motors": "EICHERMOT.NS",
        "Ashok Leyland": "ASHOKLEY.NS",
        "TVS Motor": "TVSMOTOR.NS",
    },
    "FMCG": {
        "ITC": "ITC.NS",
        "Hindustan Unilever": "HINDUNILVR.NS",
        "Nestle India": "NESTLEIND.NS",
        "Britannia Industries": "BRITANNIA.NS",
        "Tata Consumer Products": "TATACONSUM.NS",
        "Godrej Consumer Products": "GODREJCP.NS",
    },
    "Pharmaceuticals": {
        "Sun Pharma": "SUNPHARMA.NS",
        "Dr Reddy's": "DRREDDY.NS",
        "Cipla": "CIPLA.NS",
        "Divi's Laboratories": "DIVISLAB.NS",
        "Lupin": "LUPIN.NS",
        "Apollo Hospitals": "APOLLOHOSP.NS",
    },
    "Metals & Mining": {
        "Tata Steel": "TATASTEEL.NS",
        "JSW Steel": "JSWSTEEL.NS",
        "Hindalco": "HINDALCO.NS",
        "Coal India": "COALINDIA.NS",
        "Vedanta": "VEDL.NS",
        "NMDC": "NMDC.NS",
    },
    "Infrastructure & Industrials": {
        "Larsen & Toubro": "LT.NS",
        "Siemens": "SIEMENS.NS",
        "ABB India": "ABB.NS",
        "Adani Enterprises": "ADANIENT.NS",
        "Adani Ports": "ADANIPORTS.NS",
        "Bharat Electronics": "BEL.NS",
        "HAL": "HAL.NS",
        "BHEL": "BHEL.NS",
    },
    "Consumer & Retail": {
        "Titan": "TITAN.NS",
        "Trent": "TRENT.NS",
        "Avenue Supermarts": "DMART.NS",
        "Asian Paints": "ASIANPAINT.NS",
        "Pidilite Industries": "PIDILITIND.NS",
    },
    "Gold & Commodities": {
        "GoldBeES": "GOLDBEES.NS",
        "Nifty 50 BeES": "NIFTYBEES.NS",
        "SBI Gold ETF": "SETFGOLD.NS",
        "Nippon India Silver ETF": "SILVERBEES.NS",
    },
}

RATIO_LABELS = {
    "trailingPE": "P/E",
    "priceToBook": "Price / Book",
    "enterpriseToEbitda": "EV / EBITDA",
    "debtToEquity": "Debt / Equity",
    "returnOnEquity": "ROE",
    "forwardPE": "Forward P/E",
    "priceToSalesTrailing12Months": "Price / Sales",
    "enterpriseToRevenue": "EV / Revenue",
    "pegRatio": "PEG",
    "currentRatio": "Current Ratio",
    "returnOnAssets": "ROA",
    "profitMargins": "Profit Margin",
    "operatingMargins": "Operating Margin",
    "grossMargins": "Gross Margin",
    "dividendYield": "Dividend Yield",
    "marketCap": "Market Cap",
    "epsTrailingTwelveMonths": "EPS",
}
DEFAULT_RATIOS = [
    "trailingPE",
    "priceToBook",
    "enterpriseToEbitda",
    "debtToEquity",
    "returnOnEquity",
]

OPTIONS = [
    {"label": f"{name} — {ticker}", "value": ticker}
    for group in TICKERS.values()
    for name, ticker in group.items()
]

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

_HISTORY_CACHE = {}


def normalise_history(data):
    if data is None or data.empty:
        raise ValueError("Yahoo Finance returned no data.")

    data = data.copy()

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            data.columns = data.columns.get_level_values(0)
        elif "Close" in data.columns.get_level_values(-1):
            data.columns = data.columns.get_level_values(-1)
        else:
            data.columns = data.columns.get_level_values(0)

    idx = pd.to_datetime(data.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    data.index = idx.normalize()

    data = data[~data.index.duplicated(keep="last")].sort_index()
    return data.dropna(how="all")


def get_history(ticker):
    ticker = str(ticker).strip().upper()

    if ticker in _HISTORY_CACHE:
        return _HISTORY_CACHE[ticker].copy()

    errors = []

    try:
        data = yf.Ticker(ticker).history(
            period="max",
            interval="1d",
            auto_adjust=False,
            actions=False,
        )
        if data is not None and not data.empty:
            data = normalise_history(data)
            _HISTORY_CACHE[ticker] = data
            return data.copy()
    except Exception as exc:
        errors.append(str(exc))

    try:
        end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        data = yf.download(
            ticker,
            start="1900-01-01",
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="column",
        )
        data = normalise_history(data)
        _HISTORY_CACHE[ticker] = data
        return data.copy()
    except Exception as exc:
        errors.append(str(exc))

    raise ValueError(f"Could not download {ticker}: {' | '.join(errors)}")


def get_price_series(data, metric):
    if metric == "Adjusted Close" and "Adj Close" in data.columns:
        column = "Adj Close"
    elif "Close" in data.columns:
        column = "Close"
    elif "Adj Close" in data.columns:
        column = "Adj Close"
    else:
        raise ValueError("No closing-price column was returned by Yahoo Finance.")

    series = pd.to_numeric(data[column], errors="coerce").dropna()
    return series, column


def format_ratio(key, value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"

    if key in {
        "returnOnEquity",
        "returnOnAssets",
        "profitMargins",
        "operatingMargins",
        "grossMargins",
        "dividendYield",
    }:
        return f"{float(value) * 100:.2f}%"

    if key == "marketCap":
        value = float(value)
        if value >= 1e12:
            return f"{value / 1e12:.2f}T"
        if value >= 1e9:
            return f"{value / 1e9:.2f}B"
        if value >= 1e6:
            return f"{value / 1e6:.2f}M"
        return f"{value:,.0f}"

    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def get_ratios(tickers, selected):
    rows = []

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
        except Exception:
            info = {}

        row = {"Stock": ticker}
        for key in selected:
            row[RATIO_LABELS[key]] = format_ratio(key, info.get(key))
        rows.append(row)

    return rows


def calculate_stats(series, benchmark):
    series = pd.Series(series).dropna()

    if len(series) == 0:
        return {"Mean": np.nan, "Median": np.nan, "Std Dev": np.nan,
                "Positive": 0, "Observations": 0, "T-test p": np.nan,
                "Wilcoxon p": np.nan}

    t_p = np.nan
    w_p = np.nan

    if len(series) >= 2:
        try:
            t_stat, two_sided = ttest_1samp(series, popmean=benchmark)
            t_p = two_sided / 2 if t_stat > 0 else 1 - two_sided / 2
        except Exception:
            pass

        differences = series - benchmark
        differences = differences[differences != 0]
        if len(differences) >= 1:
            try:
                _, two_sided = wilcoxon(differences)
                w_p = two_sided / 2 if differences.median() > 0 else 1 - two_sided / 2
            except Exception:
                pass

    return {
        "Mean": series.mean(),
        "Median": series.median(),
        "Std Dev": series.std(ddof=1) if len(series) > 1 else np.nan,
        "Positive": int((series > 0).sum()),
        "Observations": len(series),
        "T-test p": t_p,
        "Wilcoxon p": w_p,
    }


def empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------

app = Dash(__name__, title="Stock Analytics Dashboard")
server = app.server

app.layout = html.Div(
    className="page",
    children=[
        html.H1("Stock Analytics Dashboard"),
        html.P(
            "Compare up to 10 stocks, inspect historical performance, "
            "run basic statistical tests, and view common financial ratios.",
            className="subtitle",
        ),

        html.Div(
            className="control-panel",
            children=[
                html.Label("Stocks"),
                dcc.Dropdown(
                    id="ticker",
                    options=OPTIONS,
                    value=["GOLDBEES.NS"],
                    multi=True,
                    maxHeight=300,
                    placeholder="Select up to 10 stocks",
                ),

                html.Label("Custom ticker (optional)", className="control-label"),
                dcc.Input(
                    id="custom-ticker",
                    type="text",
                    placeholder="e.g. RELIANCE.NS",
                    className="text-input",
                ),

                html.Label("Start date", className="control-label"),
                dcc.DatePickerSingle(
                    id="start-date",
                    display_format="DD-MMM-YYYY",
                    placeholder="Start date",
                    clearable=True,
                    with_portal=True,
                    number_of_months_shown=2,
                    first_day_of_week=1,
                ),

                html.Label("End date", className="control-label"),
                dcc.DatePickerSingle(
                    id="end-date",
                    display_format="DD-MMM-YYYY",
                    placeholder="End date",
                    clearable=True,
                    with_portal=True,
                    number_of_months_shown=2,
                    first_day_of_week=1,
                ),

                html.Label("Price", className="control-label"),
                dcc.Dropdown(
                    id="metric",
                    options=[
                        {"label": "Close", "value": "Close"},
                        {"label": "Adjusted Close", "value": "Adjusted Close"},
                    ],
                    value="Close",
                    clearable=False,
                ),

                html.Label("Rolling mean window", className="control-label"),
                dcc.Input(
                    id="window",
                    type="number",
                    min=1,
                    step=1,
                    value=3,
                    className="number-input",
                ),

                html.Label("Benchmark (%)", className="control-label"),
                dcc.Input(
                    id="benchmark",
                    type="number",
                    value=1.0,
                    step=0.1,
                    className="number-input",
                ),

                html.Label("Chart", className="control-label"),
                dcc.Dropdown(
                    id="chart-type",
                    options=[
                        {"label": "Line", "value": "line"},
                        {"label": "Candlestick", "value": "candlestick"},
                        {"label": "OHLC", "value": "ohlc"},
                    ],
                    value="line",
                    clearable=False,
                ),

                html.Label("Financial ratios", className="control-label"),
                dcc.Dropdown(
                    id="ratio-selection",
                    options=[
                        {"label": label, "value": key}
                        for key, label in RATIO_LABELS.items()
                    ],
                    value=DEFAULT_RATIOS,
                    multi=True,
                    clearable=False,
                ),

                html.Div(
                    className="buttons",
                    children=[
                        html.Button("Analyse", id="analyse", n_clicks=0, className="analyse"),
                        html.Button("Reset", id="reset", n_clicks=0, className="reset"),
                    ],
                ),
                html.Div(id="status", className="status"),
            ],
        ),

        html.Div(
            className="chart-grid",
            children=[
                dcc.Graph(id="price-chart"),
                dcc.Graph(id="indexed-chart"),
            ],
        ),

        html.Div(
            className="results-panel",
            children=[
                html.H2("Results"),
                dash_table.DataTable(
                    id="results-table",
                    columns=[],
                    data=[],
                    page_size=15,
                    sort_action="native",
                    style_table={"overflowX": "auto"},
                    style_cell={"padding": "8px", "textAlign": "left"},
                    style_header={"fontWeight": "bold"},
                ),
                html.H2("Financial Ratios"),
                dash_table.DataTable(
                    id="ratio-table",
                    columns=[],
                    data=[],
                    page_size=10,
                    style_table={"overflowX": "auto"},
                    style_cell={"padding": "8px", "textAlign": "left"},
                    style_header={"fontWeight": "bold"},
                ),
                html.H2("Statistical analysis"),
                html.Div(id="statistics"),
            ],
        ),
    ],
)


# Reset only changes controls. It does not contain hidden date logic.
@app.callback(
    Output("ticker", "value"),
    Output("custom-ticker", "value"),
    Output("metric", "value"),
    Output("window", "value"),
    Output("benchmark", "value"),
    Output("chart-type", "value"),
    Output("ratio-selection", "value"),
    Input("reset", "n_clicks"),
    prevent_initial_call=True,
)
def reset_controls(_):
    return (
        ["GOLDBEES.NS"],
        "",
        "Close",
        3,
        1.0,
        "line",
        DEFAULT_RATIOS,
    )


@app.callback(
    Output("price-chart", "figure"),
    Output("indexed-chart", "figure"),
    Output("results-table", "columns"),
    Output("results-table", "data"),
    Output("ratio-table", "columns"),
    Output("ratio-table", "data"),
    Output("statistics", "children"),
    Output("status", "children"),
    Input("analyse", "n_clicks"),
    State("ticker", "value"),
    State("custom-ticker", "value"),
    State("start-date", "date"),
    State("end-date", "date"),
    State("metric", "value"),
    State("window", "value"),
    State("benchmark", "value"),
    State("chart-type", "value"),
    State("ratio-selection", "value"),
    prevent_initial_call=True,
)
def analyse(
    _clicks,
    tickers,
    custom_ticker,
    start_date,
    end_date,
    metric,
    window,
    benchmark,
    chart_type,
    ratio_selection,
):
    try:
        selected = [x for x in (tickers or []) if x]

        if custom_ticker and custom_ticker.strip():
            custom = custom_ticker.strip().upper()
            if custom not in selected:
                selected.append(custom)

        if not selected:
            raise ValueError("Select at least one stock.")

        if len(selected) > 10:
            raise ValueError("You can analyse a maximum of 10 stocks.")

        if not start_date or not end_date:
            raise ValueError("Enter both a start date and an end date.")

        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()

        if start > end:
            raise ValueError("Start date must be before end date.")

        window = max(1, int(window))
        benchmark = float(benchmark)

        histories = {}
        series_map = {}

        for ticker in selected:
            history = get_history(ticker)
            histories[ticker] = history

            series, _ = get_price_series(history, metric)
            series = series.loc[(series.index >= start) & (series.index <= end)]

            if series.empty:
                raise ValueError(
                    f"No historical data for {ticker} between "
                    f"{start.date()} and {end.date()}."
                )

            series_map[ticker] = series

        # Price chart.
        price_fig = go.Figure()

        for ticker, series in series_map.items():
            if chart_type == "line":
                price_fig.add_trace(
                    go.Scatter(
                        x=series.index,
                        y=series.values,
                        mode="lines",
                        name=ticker,
                    )
                )
            else:
                h = histories[ticker].loc[start:end]
                if chart_type == "candlestick":
                    price_fig.add_trace(
                        go.Candlestick(
                            x=h.index,
                            open=h["Open"],
                            high=h["High"],
                            low=h["Low"],
                            close=h["Close"],
                            name=ticker,
                        )
                    )
                else:
                    price_fig.add_trace(
                        go.Ohlc(
                            x=h.index,
                            open=h["Open"],
                            high=h["High"],
                            low=h["Low"],
                            close=h["Close"],
                            name=ticker,
                        )
                    )

        price_fig.update_layout(
            title="Historical Price",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified",
            xaxis_rangeslider_visible=(chart_type == "candlestick"),
        )

        # Indexed chart: every stock starts at 100.
        indexed_fig = go.Figure()
        result_rows = []

        for ticker, series in series_map.items():
            first = float(series.iloc[0])
            indexed = series / first * 100.0

            indexed_fig.add_trace(
                go.Scatter(
                    x=indexed.index,
                    y=indexed.values,
                    mode="lines",
                    name=ticker,
                )
            )

            last = float(series.iloc[-1])
            total_return = (last / first - 1) * 100

            rolling_values = series.rolling(window=window, min_periods=window).mean().dropna()
            rolling_return = np.nan
            if len(rolling_values) >= 2:
                rolling_return = (
                    rolling_values.iloc[-1] / rolling_values.iloc[0] - 1
                ) * 100

            result_rows.append(
                {
                    "Stock": ticker,
                    "Start Price": round(first, 2),
                    "End Price": round(last, 2),
                    "Return (%)": round(total_return, 2),
                    f"{window}-Day Rolling Mean Return (%)": (
                        round(rolling_return, 2)
                        if not np.isnan(rolling_return)
                        else "N/A"
                    ),
                    "Positive": "Yes" if total_return > 0 else "No",
                }
            )

        indexed_fig.update_layout(
            title="Indexed Performance (Start = 100)",
            xaxis_title="Date",
            yaxis_title="Indexed Value",
            hovermode="x unified",
        )

        # Ratios.
        ratio_rows = get_ratios(selected, ratio_selection or DEFAULT_RATIOS)
        ratio_columns = (
            [{"name": c, "id": c} for c in ratio_rows[0].keys()]
            if ratio_rows
            else []
        )

        # Statistical tests only make sense for one selected investment.
        if len(selected) == 1:
            ticker = selected[0]
            series = series_map[ticker]
            returns = series.pct_change().dropna() * 100

            stats = calculate_stats(returns, benchmark)

            statistics = html.Div(
                className="stats-grid",
                children=[
                    html.Div(f"Mean daily return: {stats['Mean']:.3f}%"),
                    html.Div(f"Median daily return: {stats['Median']:.3f}%"),
                    html.Div(f"Std deviation: {stats['Std Dev']:.3f}%"),
                    html.Div(f"Positive days: {stats['Positive']}"),
                    html.Div(f"Observations: {stats['Observations']}"),
                    html.Div(
                        f"One-sided t-test p-value: "
                        f"{stats['T-test p']:.4f}"
                        if not np.isnan(stats["T-test p"])
                        else "One-sided t-test: unavailable"
                    ),
                    html.Div(
                        f"One-sided Wilcoxon p-value: "
                        f"{stats['Wilcoxon p']:.4f}"
                        if not np.isnan(stats["Wilcoxon p"])
                        else "One-sided Wilcoxon: unavailable"
                    ),
                ],
            )
        else:
            best = max(
                result_rows,
                key=lambda row: row["Return (%)"],
            )
            statistics = html.Div(
                [
                    html.P(
                        "Statistical tests are available only when exactly one "
                        "stock is selected."
                    ),
                    html.P(
                        f"Best historical return in this period: "
                        f"{best['Stock']} ({best['Return (%)']:.2f}%)."
                    ),
                ]
            )

        status = (
            f"Analysed {len(selected)} stock(s) from "
            f"{start.date()} to {end.date()}."
        )

        result_columns = [
            {"name": c, "id": c} for c in result_rows[0].keys()
        ]

        return (
            price_fig,
            indexed_fig,
            result_columns,
            result_rows,
            ratio_columns,
            ratio_rows,
            statistics,
            status,
        )

    except Exception as exc:
        error = str(exc)
        return (
            empty_figure("Analysis error"),
            empty_figure("Analysis error"),
            [],
            [],
            [],
            [],
            html.Div(f"Error: {error}", className="error"),
            f"Error: {error}",
        )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)

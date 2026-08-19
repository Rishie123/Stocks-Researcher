
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import ttest_1samp, wilcoxon
from dash import Dash, Input, Output, State, dcc, html, dash_table
import plotly.graph_objects as go

# ---------------------------- Stock universe ---------------------------- #

STOCKS = {
    "NIFTY 50": "^NSEI", "NIFTY BANK": "^NSEBANK",
    "NIFTY IT": "^CNXIT", "NIFTY AUTO": "^CNXAUTO",
    "NIFTY PHARMA": "^CNXPHARMA", "NIFTY FMCG": "^CNXFMCG",
    "NIFTY METAL": "^CNXMETAL",
    "HDFC Bank": "HDFCBANK.NS", "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS", "Axis Bank": "AXISBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS", "IndusInd Bank": "INDUSINDBK.NS",
    "Bank of Baroda": "BANKBARODA.NS", "Punjab National Bank": "PNB.NS",
    "Canara Bank": "CANBK.NS", "Union Bank of India": "UNIONBANK.NS",
    "Bajaj Finance": "BAJFINANCE.NS", "Bajaj Finserv": "BAJAJFINSV.NS",
    "Shriram Finance": "SHRIRAMFIN.NS",
    "TCS": "TCS.NS", "Infosys": "INFY.NS",
    "HCL Technologies": "HCLTECH.NS", "Wipro": "WIPRO.NS",
    "Tech Mahindra": "TECHM.NS", "LTIMindtree": "LTIM.NS",
    "Persistent Systems": "PERSISTENT.NS",
    "Reliance Industries": "RELIANCE.NS", "ONGC": "ONGC.NS",
    "Indian Oil": "IOC.NS", "BPCL": "BPCL.NS", "GAIL": "GAIL.NS",
    "NTPC": "NTPC.NS", "Power Grid": "POWERGRID.NS",
    "Maruti Suzuki": "MARUTI.NS", "Tata Motors": "TATAMOTORS.NS",
    "Mahindra & Mahindra": "M&M.NS", "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS", "Eicher Motors": "EICHERMOT.NS",
    "Ashok Leyland": "ASHOKLEY.NS", "TVS Motor": "TVSMOTOR.NS",
    "ITC": "ITC.NS", "Hindustan Unilever": "HINDUNILVR.NS",
    "Nestle India": "NESTLEIND.NS", "Britannia Industries": "BRITANNIA.NS",
    "Tata Consumer Products": "TATACONSUM.NS",
    "Sun Pharma": "SUNPHARMA.NS", "Dr Reddy's": "DRREDDY.NS",
    "Cipla": "CIPLA.NS", "Divi's Laboratories": "DIVISLAB.NS",
    "Lupin": "LUPIN.NS", "Tata Steel": "TATASTEEL.NS",
    "JSW Steel": "JSWSTEEL.NS", "Hindalco": "HINDALCO.NS",
    "Coal India": "COALINDIA.NS", "Vedanta": "VEDL.NS",
    "Larsen & Toubro": "LT.NS", "Siemens": "SIEMENS.NS",
    "ABB India": "ABB.NS", "Adani Enterprises": "ADANIENT.NS",
    "Adani Ports": "ADANIPORTS.NS", "Bharat Electronics": "BEL.NS",
    "HAL": "HAL.NS", "BHEL": "BHEL.NS", "Titan": "TITAN.NS",
    "Trent": "TRENT.NS", "Avenue Supermarts": "DMART.NS",
    "Asian Paints": "ASIANPAINT.NS", "Pidilite Industries": "PIDILITIND.NS",
    "GoldBeES": "GOLDBEES.NS", "Nifty 50 BeES": "NIFTYBEES.NS",
    "SBI Gold ETF": "SETFGOLD.NS", "Nippon India Silver ETF": "SILVERBEES.NS",
}

RATIOS = {
    "trailingPE": "P/E", "priceToBook": "Price / Book",
    "enterpriseToEbitda": "EV / EBITDA", "debtToEquity": "Debt / Equity",
    "returnOnEquity": "ROE", "forwardPE": "Forward P/E",
    "priceToSalesTrailing12Months": "Price / Sales",
    "enterpriseToRevenue": "EV / Revenue", "pegRatio": "PEG",
    "currentRatio": "Current Ratio", "returnOnAssets": "ROA",
    "profitMargins": "Profit Margin", "operatingMargins": "Operating Margin",
    "grossMargins": "Gross Margin", "dividendYield": "Dividend Yield",
    "marketCap": "Market Cap", "epsTrailingTwelveMonths": "EPS",
}
DEFAULT_RATIOS = ["trailingPE", "priceToBook", "enterpriseToEbitda",
                  "debtToEquity", "returnOnEquity"]

OPTIONS = [{"label": name, "value": ticker} for name, ticker in STOCKS.items()]
CACHE = {}


# ---------------------------- Data functions ---------------------------- #

def clean_history(data):
    if data is None or data.empty:
        raise ValueError("Yahoo Finance returned no data.")
    data = data.copy()
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            data.columns = data.columns.get_level_values(0)
        else:
            data.columns = data.columns.get_level_values(-1)
    idx = pd.to_datetime(data.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    data.index = idx.normalize()
    return data[~data.index.duplicated(keep="last")].sort_index()


def history(ticker):
    ticker = ticker.strip().upper()
    if ticker in CACHE:
        return CACHE[ticker].copy()

    try:
        data = yf.Ticker(ticker).history(
            period="max", interval="1d", auto_adjust=False, actions=False
        )
        data = clean_history(data)
    except Exception as first:
        try:
            end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            data = yf.download(
                ticker, start="1900-01-01", end=end, interval="1d",
                auto_adjust=False, progress=False, threads=False
            )
            data = clean_history(data)
        except Exception as second:
            raise ValueError(f"Could not retrieve {ticker}: {first}; {second}")

    if "Close" not in data.columns:
        raise ValueError(f"No closing-price data returned for {ticker}.")
    CACHE[ticker] = data
    return data.copy()


def ratio_text(key, value):
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(value):
        return "N/A"
    if key in {"returnOnEquity", "returnOnAssets", "profitMargins",
               "operatingMargins", "grossMargins", "dividendYield"}:
        return f"{value * 100:.2f}%"
    if key == "marketCap":
        if value >= 1e12:
            return f"{value / 1e12:.2f}T"
        if value >= 1e9:
            return f"{value / 1e9:.2f}B"
        if value >= 1e6:
            return f"{value / 1e6:.2f}M"
    return f"{value:.2f}"


def financial_ratios(tickers, selected):
    rows = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
        except Exception:
            info = {}
        row = {"Stock": ticker}
        for key in selected:
            row[RATIOS[key]] = ratio_text(key, info.get(key))
        rows.append(row)
    return rows


def tests(returns, benchmark):
    returns = pd.Series(returns).dropna()
    out = {
        "mean": returns.mean(), "median": returns.median(),
        "std": returns.std(ddof=1) if len(returns) > 1 else np.nan,
        "positive": int((returns > 0).sum()), "n": len(returns),
        "ttest": np.nan, "wilcoxon": np.nan,
    }
    if len(returns) >= 2:
        try:
            stat, p = ttest_1samp(returns, benchmark)
            out["ttest"] = p / 2 if stat > 0 else 1 - p / 2
        except Exception:
            pass
        d = returns - benchmark
        d = d[d != 0]
        if len(d):
            try:
                _, p = wilcoxon(d)
                out["wilcoxon"] = p / 2 if d.median() > 0 else 1 - p / 2
            except Exception:
                pass
    return out


def blank(message):
    fig = go.Figure()
    fig.add_annotation(text=message, x=.5, y=.5, xref="paper", yref="paper",
                       showarrow=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


# ------------------------------- App ----------------------------------- #

app = Dash(__name__, title="Stock Analytics")
server = app.server

app.layout = html.Div(className="page", children=[
    html.Div(className="hero", children=[
        html.Div("STOCK ANALYTICS", className="eyebrow"),
        html.H1("Compare. Analyse. Decide."),
        html.P("Historical returns, return distributions, statistical evidence "
               "and financial ratios — without unnecessary price clutter.")
    ]),

    html.Div(className="workspace", children=[
        html.Div(className="controls", children=[
            html.H2("Analysis"),
            html.P("Select up to 10 investments and an exact period."),

            html.Label("Stocks"),
            dcc.Dropdown(id="stocks", options=OPTIONS, value=["GOLDBEES.NS"],
                         multi=True, maxHeight=280,
                         placeholder="Select up to 10 stocks"),

            html.Label("Custom ticker"),
            dcc.Input(id="custom", type="text",
                      placeholder="Optional: RELIANCE.NS", className="input"),

            html.Div(className="two-col", children=[
                html.Div([
                    html.Label("Start date"),
                    dcc.DatePickerSingle(id="start-date",
                                         display_format="DD-MMM-YYYY",
                                         placeholder="Start date",
                                         clearable=True, with_portal=True,
                                         number_of_months_shown=2)
                ]),
                html.Div([
                    html.Label("End date"),
                    dcc.DatePickerSingle(id="end-date",
                                         display_format="DD-MMM-YYYY",
                                         placeholder="End date",
                                         clearable=True, with_portal=True,
                                         number_of_months_shown=2)
                ])
            ]),

            html.Div(className="two-col", children=[
                html.Div([
                    html.Label("Rolling mean"),
                    dcc.Input(id="window", type="number", min=1, step=1,
                              value=3, className="input")
                ]),
                html.Div([
                    html.Label("Benchmark (%)"),
                    dcc.Input(id="benchmark", type="number", step=.1,
                              value=1.0, className="input")
                ])
            ]),

            html.Label("Financial ratios"),
            dcc.Dropdown(
                id="ratios",
                options=[{"label": label, "value": key}
                         for key, label in RATIOS.items()],
                value=DEFAULT_RATIOS, multi=True, clearable=False
            ),

            html.Div(className="actions", children=[
                html.Button("Analyse", id="analyse", n_clicks=0,
                            className="primary"),
                html.Button("Reset", id="reset", n_clicks=0,
                            className="secondary")
            ]),
            html.Div(id="status", className="status")
        ]),

        html.Div(className="content", children=[
            html.Div(className="chart-card", children=[
                html.Div([
                    html.H2("Cumulative return"),
                    html.P("Each selected investment starts at 0% for an easy comparison.")
                ], className="heading"),
                dcc.Graph(id="return-chart",
                          config={"displaylogo": False, "responsive": True},
                          style={"height": "min(72vh, 820px)"})
            ]),

            html.Div(className="chart-card", children=[
                html.Div([
                    html.H2("Candlestick chart"),
                    html.P("Daily OHLC price action for the selected stock.")
                ], className="heading"),
                dcc.Graph(
                    id="candlestick-chart",
                    config={"displaylogo": False, "responsive": True},
                    style={"height": "min(72vh, 820px)"}
                )
            ]),

            html.Div(className="card", children=[
                html.H2("Return summary"),
                dash_table.DataTable(
                    id="results", columns=[], data=[], page_size=10,
                    sort_action="native", style_table={"overflowX": "auto"},
                    style_cell={"padding": "10px", "minWidth": "110px"},
                    style_header={"fontWeight": "700"})
            ]),

            html.Div(className="card", children=[
                html.H2("Financial ratios"),
                dash_table.DataTable(
                    id="ratio-table", columns=[], data=[], page_size=10,
                    style_table={"overflowX": "auto"},
                    style_cell={"padding": "10px", "minWidth": "110px"},
                    style_header={"fontWeight": "700"})
            ]),

            html.Div(className="card", children=[
                html.H2("Statistical analysis"),
                html.Div(id="statistics")
            ])
        ])
    ])
])


@app.callback(
    Output("stocks", "value"), Output("custom", "value"),
    Output("window", "value"), Output("benchmark", "value"),
    Output("ratios", "value"),
    Input("reset", "n_clicks"), prevent_initial_call=True
)
def reset(_):
    return ["GOLDBEES.NS"], "", 3, 1.0, DEFAULT_RATIOS


@app.callback(
    Output("return-chart", "figure"),
    Output("candlestick-chart", "figure"),
    Output("results", "columns"), Output("results", "data"),
    Output("ratio-table", "columns"), Output("ratio-table", "data"),
    Output("statistics", "children"), Output("status", "children"),
    Input("analyse", "n_clicks"),
    State("stocks", "value"), State("custom", "value"),
    State("start-date", "date"), State("end-date", "date"),
    State("window", "value"), State("benchmark", "value"),
    State("ratios", "value"), prevent_initial_call=True
)
def analyse(_, selected, custom, start_date, end_date, window, benchmark, selected_ratios):
    try:
        tickers = list(dict.fromkeys(selected or []))
        if custom and custom.strip():
            custom = custom.strip().upper()
            if custom not in tickers:
                tickers.append(custom)

        if not tickers:
            raise ValueError("Select at least one stock.")
        if len(tickers) > 10:
            raise ValueError("A maximum of 10 stocks can be analysed.")
        if not start_date or not end_date:
            raise ValueError("Enter both a start date and an end date.")

        start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
        if start > end:
            raise ValueError("Start date must be before end date.")

        window = max(1, int(window))
        benchmark = float(benchmark)

        series_map = {}
        notes = []

        for ticker in tickers:
            data = history(ticker)
            prices = data["Close"].dropna()
            lo, hi = prices.index.min(), prices.index.max()

            actual_start, actual_end = max(start, lo), min(end, hi)
            if start < lo:
                notes.append(f"{ticker} starts on {lo.strftime('%d-%b-%Y')}.")
            if end > hi:
                notes.append(f"{ticker} ends on {hi.strftime('%d-%b-%Y')}.")

            series = prices.loc[(prices.index >= actual_start) &
                                (prices.index <= actual_end)]
            if series.empty:
                raise ValueError(f"No data for {ticker} in the requested period.")
            series_map[ticker] = series

        # Bottom visual: candlestick chart for the first selected stock.
        # This is intentionally shown by default rather than asking the user
        # to choose a raw-price chart mode.
        candle_ticker = tickers[0]
        candle_data = history(candle_ticker)
        candle_series = series_map[candle_ticker]
        candle_data = candle_data.loc[
            candle_data.index.intersection(candle_series.index)
        ]

        candle_fig = go.Figure()
        if all(col in candle_data.columns for col in ["Open", "High", "Low", "Close"]):
            candle_fig.add_trace(go.Candlestick(
                x=candle_data.index,
                open=candle_data["Open"],
                high=candle_data["High"],
                low=candle_data["Low"],
                close=candle_data["Close"],
                name=candle_ticker,
            ))
        else:
            candle_fig.add_annotation(
                text=f"OHLC data unavailable for {candle_ticker}",
                x=.5, y=.5, xref="paper", yref="paper",
                showarrow=False
            )

        candle_fig.update_layout(
            title=f"{candle_ticker} — Candlestick",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            margin=dict(l=65, r=25, t=85, b=65),
            hovermode="x unified",
        )

        # Primary visual: cumulative returns.
        return_fig = go.Figure()
        rows = []

        for ticker, series in series_map.items():
            cumulative = (series / series.iloc[0] - 1) * 100
            return_fig.add_trace(go.Scatter(
                x=series.index, y=cumulative, mode="lines", name=ticker))

            daily = series.pct_change().dropna() * 100
            rolling = series.rolling(window, min_periods=window).mean().dropna()
            rolling_return = np.nan
            if len(rolling) >= 2:
                rolling_return = (rolling.iloc[-1] / rolling.iloc[0] - 1) * 100

            rows.append({
                "Stock": ticker,
                "Return (%)": round(float(cumulative.iloc[-1]), 2),
                f"{window}-Day Rolling Return (%)":
                    round(float(rolling_return), 2)
                    if not np.isnan(rolling_return) else "N/A",
                "Daily Volatility (%)":
                    round(float(daily.std()), 2) if len(daily) > 1 else "N/A",
                "Positive Days": int((daily > 0).sum()),
                "Trading Days": int(len(daily))
            })

        return_fig.add_hline(y=0, line_width=1)
        return_fig.update_layout(
            title=dict(
                text="Cumulative Return",
                x=0.02,
                xanchor="left",
                y=0.98,
                yanchor="top",
            ),
            yaxis_title="Return (%)",
            xaxis_title="Date",
            hovermode="x unified",
            margin=dict(l=65, r=25, t=125, b=65),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.08,
                xanchor="left",
                x=0,
                entrywidth=170,
                entrywidthmode="pixels",
            ),
        )

        result_cols = [{"name": c, "id": c} for c in rows[0].keys()]
        ratio_rows = financial_ratios(tickers, selected_ratios or DEFAULT_RATIOS)
        ratio_cols = ([{"name": c, "id": c} for c in ratio_rows[0].keys()]
                      if ratio_rows else [])

        if len(tickers) == 1:
            returns = series_map[tickers[0]].pct_change().dropna() * 100
            s = tests(returns, benchmark)
            statistics = html.Div(className="stats", children=[
                html.Div([html.Span("Mean daily return"), html.Strong(f"{s['mean']:.3f}%")]),
                html.Div([html.Span("Median daily return"), html.Strong(f"{s['median']:.3f}%")]),
                html.Div([html.Span("Daily volatility"), html.Strong(f"{s['std']:.3f}%")]),
                html.Div([html.Span("Positive days"), html.Strong(str(s["positive"]))]),
                html.Div([html.Span("Observations"), html.Strong(str(s["n"]))]),
                html.Div([html.Span("T-test p-value"),
                          html.Strong(f"{s['ttest']:.4f}" if not np.isnan(s["ttest"]) else "Unavailable")]),
                html.Div([html.Span("Wilcoxon p-value"),
                          html.Strong(f"{s['wilcoxon']:.4f}" if not np.isnan(s["wilcoxon"]) else "Unavailable")])
            ])
        else:
            best = max(rows, key=lambda r: r["Return (%)"])
            statistics = html.Div([
                html.P("Statistical tests are available when exactly one stock is selected."),
                html.P(f"Highest return in this period: {best['Stock']} ({best['Return (%)']:.2f}%).")
            ])

        status = f"Analysed {len(tickers)} stock(s): {start.strftime('%d-%b-%Y')} to {end.strftime('%d-%b-%Y')}."
        if notes:
            status += " " + " ".join(notes)

        return (return_fig, candle_fig, result_cols, rows,
                ratio_cols, ratio_rows, statistics, status)

    except Exception as exc:
        msg = f"Analysis error: {exc}"
        return (blank(msg), blank(msg), [], [], [], [],
                html.Div(msg, className="error"), msg)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)

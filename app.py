import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import ttest_1samp, wilcoxon
from dash import Dash, Input, Output, State, dcc, html, dash_table
import plotly.graph_objects as go

TICKERS = {
    "Indices": {
        "NIFTY 50": "^NSEI", "NIFTY BANK": "^NSEBANK", "NIFTY IT": "^CNXIT",
        "NIFTY AUTO": "^CNXAUTO", "NIFTY PHARMA": "^CNXPHARMA",
        "NIFTY FMCG": "^CNXFMCG", "NIFTY METAL": "^CNXMETAL",
    },
    "Banking & Financial": {
        "HDFC Bank": "HDFCBANK.NS", "ICICI Bank": "ICICIBANK.NS",
        "State Bank of India": "SBIN.NS", "Axis Bank": "AXISBANK.NS",
        "Kotak Mahindra Bank": "KOTAKBANK.NS", "IndusInd Bank": "INDUSINDBK.NS",
        "Bank of Baroda": "BANKBARODA.NS", "Punjab National Bank": "PNB.NS",
        "Canara Bank": "CANBK.NS", "Union Bank of India": "UNIONBANK.NS",
        "Bajaj Finance": "BAJFINANCE.NS", "Bajaj Finserv": "BAJAJFINSV.NS",
        "Shriram Finance": "SHRIRAMFIN.NS", "SBI Life": "SBILIFE.NS",
        "HDFC Life": "HDFCLIFE.NS",
    },
    "IT": {
        "TCS": "TCS.NS", "Infosys": "INFY.NS", "HCL Technologies": "HCLTECH.NS",
        "Wipro": "WIPRO.NS", "Tech Mahindra": "TECHM.NS",
        "LTIMindtree": "LTIM.NS", "Persistent Systems": "PERSISTENT.NS",
        "Mphasis": "MPHASIS.NS",
    },
    "Energy & Oil": {
        "Reliance Industries": "RELIANCE.NS", "ONGC": "ONGC.NS",
        "Indian Oil": "IOC.NS", "BPCL": "BPCL.NS",
        "Hindustan Petroleum": "HINDPETRO.NS", "GAIL": "GAIL.NS",
        "Adani Green Energy": "ADANIGREEN.NS", "NTPC": "NTPC.NS",
        "Power Grid": "POWERGRID.NS",
    },
    "Automobile": {
        "Maruti Suzuki": "MARUTI.NS", "Tata Motors": "TATAMOTORS.NS",
        "Mahindra & Mahindra": "M&M.NS", "Bajaj Auto": "BAJAJ-AUTO.NS",
        "Hero MotoCorp": "HEROMOTOCO.NS", "Eicher Motors": "EICHERMOT.NS",
        "Ashok Leyland": "ASHOKLEY.NS", "TVS Motor": "TVSMOTOR.NS",
    },
    "FMCG": {
        "ITC": "ITC.NS", "Hindustan Unilever": "HINDUNILVR.NS",
        "Nestle India": "NESTLEIND.NS", "Britannia Industries": "BRITANNIA.NS",
        "Tata Consumer Products": "TATACONSUM.NS", "Godrej Consumer Products": "GODREJCP.NS",
    },
    "Pharmaceuticals": {
        "Sun Pharma": "SUNPHARMA.NS", "Dr Reddy's": "DRREDDY.NS",
        "Cipla": "CIPLA.NS", "Divi's Laboratories": "DIVISLAB.NS",
        "Lupin": "LUPIN.NS", "Apollo Hospitals": "APOLLOHOSP.NS",
    },
    "Metals & Mining": {
        "Tata Steel": "TATASTEEL.NS", "JSW Steel": "JSWSTEEL.NS",
        "Hindalco": "HINDALCO.NS", "Coal India": "COALINDIA.NS",
        "Vedanta": "VEDL.NS", "NMDC": "NMDC.NS",
    },
    "Infrastructure & Industrials": {
        "Larsen & Toubro": "LT.NS", "Siemens": "SIEMENS.NS",
        "ABB India": "ABB.NS", "Adani Enterprises": "ADANIENT.NS",
        "Adani Ports": "ADANIPORTS.NS", "Bharat Electronics": "BEL.NS",
        "HAL": "HAL.NS", "BHEL": "BHEL.NS",
    },
    "Consumer & Retail": {
        "Titan": "TITAN.NS", "Trent": "TRENT.NS", "Avenue Supermarts": "DMART.NS",
        "Asian Paints": "ASIANPAINT.NS", "Pidilite Industries": "PIDILITIND.NS",
    },
    "Gold & Commodities": {
        "GoldBeES": "GOLDBEES.NS",
    "Nifty 50 BeES": "NIFTYBEES.NS", "SBI Gold ETF": "SETFGOLD.NS",
        "Nippon India Silver ETF": "SILVERBEES.NS",
        "Nifty 50 BeES": "NIFTYBEES.NS",
    },
}

def options():
    out = []
    for category, stocks in TICKERS.items():
        for name, ticker in stocks.items():
            out.append({"label": f"{name} — {ticker}", "value": ticker})
    return out

def download_history(ticker):
    data = yf.download(
        ticker, period="max", interval="1d",
        auto_adjust=False, progress=False, threads=False
    )
    if data is None or data.empty:
        raise ValueError(f"No historical data returned for {ticker}.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.index = pd.to_datetime(data.index).tz_localize(None).normalize()
    data = data[~data.index.duplicated(keep="last")].sort_index()
    return data.dropna(how="all")

def clean_price_series(data, metric):
    col = "Adj Close" if metric == "Adjusted Close" and "Adj Close" in data.columns else "Close"
    s = pd.to_numeric(data[col], errors="coerce").dropna()
    return s, col

def trading_value(s, requested_date, window):
    d = pd.Timestamp(requested_date).normalize()
    eligible = s.loc[s.index <= d]
    if eligible.empty:
        raise ValueError(f"No trading data on or before {d.date()}.")
    return float(eligible.iloc[-window:].mean()), eligible.index[-1]

def descriptive(returns):
    if len(returns) == 0:
        return {}
    return {
        "Mean": returns.mean(),
        "Median": returns.median(),
        "Std Dev": returns.std(ddof=1) if len(returns) > 1 else np.nan,
        "Minimum": returns.min(),
        "Maximum": returns.max(),
        "Positive Periods": int((returns > 0).sum()),
        "Negative Periods": int((returns < 0).sum()),
        "Observations": len(returns),
    }

def test_statistics(returns, benchmark):
    r = pd.Series(returns).dropna()
    if len(r) < 5:
        return None
    shifted = r - benchmark
    shifted_nonzero = shifted[shifted != 0]
    if len(shifted_nonzero) < 5:
        return None
    t_stat, t_p_two = ttest_1samp(r, popmean=benchmark)
    t_p = t_p_two / 2 if t_stat > 0 else 1 - t_p_two / 2
    try:
        _, w_p_two = wilcoxon(shifted_nonzero)
        w_p = w_p_two / 2 if shifted_nonzero.median() > 0 else 1 - w_p_two / 2
    except ValueError:
        w_p = np.nan
    return {"T-test p-value": t_p, "Wilcoxon p-value": w_p}

def card(title, value):
    return html.Div([
        html.Div(title, className="card-title"),
        html.Div(value, className="card-value")
    ], className="card")


RATIO_LABELS = {
    "trailingPE": "P/E Ratio",
    "forwardPE": "Forward P/E",
    "priceToBook": "Price / Book",
    "priceToSalesTrailing12Months": "Price / Sales",
    "enterpriseToEbitda": "EV / EBITDA",
    "enterpriseToRevenue": "EV / Revenue",
    "pegRatio": "PEG Ratio",
    "debtToEquity": "Debt / Equity",
    "currentRatio": "Current Ratio",
    "returnOnEquity": "ROE",
    "returnOnAssets": "ROA",
    "profitMargins": "Profit Margin",
    "operatingMargins": "Operating Margin",
    "grossMargins": "Gross Margin",
    "dividendYield": "Dividend Yield",
    "marketCap": "Market Cap",
    "epsTrailingTwelveMonths": "EPS",
}
DEFAULT_RATIOS = ["trailingPE","priceToBook","enterpriseToEbitda","debtToEquity","returnOnEquity"]

def get_ratios(tickers, selected):
    rows=[]
    for ticker in tickers:
        try:
            info=yf.Ticker(ticker).info
        except Exception:
            info={}
        row={"Stock":ticker}
        for key in selected:
            label=RATIO_LABELS[key]
            value=info.get(key)
            if value is None:
                row[label]="N/A"
            elif key in {"returnOnEquity","returnOnAssets","profitMargins","operatingMargins","grossMargins","dividendYield"}:
                row[label]=f"{value*100:.2f}%"
            elif key=="marketCap":
                row[label]=f"{value/1e12:.2f}T" if value>=1e12 else (f"{value/1e9:.2f}B" if value>=1e9 else f"{value:,.0f}")
            else:
                try: row[label]=f"{float(value):.2f}"
                except (TypeError,ValueError): row[label]=str(value)
        rows.append(row)
    return rows

app = Dash(__name__, title="Stock Analytics Dashboard")
server = app.server

app.layout = html.Div([
    html.Div([
        html.H1("Stock Analytics Dashboard"),
        html.P("Flexible historical analysis of stocks, ETFs and indices using Yahoo Finance.")
    ], className="hero"),

    html.Div([
        html.Div([
            html.Label("Stocks to compare (maximum 10)"),
            dcc.Dropdown(
                id="ticker", options=options(),
                value=["GOLDBEES.NS"],
                searchable=True, multi=True, clearable=False,
                placeholder="Search and select up to 10 stocks..."
            ),
            html.Label("Optional custom Yahoo ticker", className="sub-label"),
            dcc.Input(id="custom-ticker", type="text", placeholder="Optional: e.g. RELIANCE.NS",
                      style={"width": "100%"})
        ], className="control"),

        html.Div([
            html.Label("Analysis mode"),
            dcc.RadioItems(
                id="mode",
                options=[
                    {"label": " Single period", "value": "single"},
                    {"label": " Historical repeated periods", "value": "historical"},
                ],
                value="single", inline=True
            ),
            html.Label("Analysis dates", className="sub-label"),
            dcc.DatePickerRange(
                id="date-range",
                display_format="DD-MMM-YYYY",
                start_date_placeholder_text="Start date",
                end_date_placeholder_text="End date",
                clearable=False,
                minimum_nights=0,
                number_of_months_shown=2,
                first_day_of_week=1,
                with_portal=True,
                updatemode="singledate",
                day_size=36
            ),
            html.Small(
                "Full available Yahoo Finance history. You can type a date directly or navigate month-by-month.",
                style={"display":"block","opacity":".65","marginTop":"6px","marginBottom":"8px"}
            ),
            html.Label("Quick range", className="sub-label"),
            html.Div([
                html.Button("1M", id="range-1m", n_clicks=0, className="range-button"),
                html.Button("3M", id="range-3m", n_clicks=0, className="range-button"),
                html.Button("6M", id="range-6m", n_clicks=0, className="range-button"),
                html.Button("1Y", id="range-1y", n_clicks=0, className="range-button"),
                html.Button("5Y", id="range-5y", n_clicks=0, className="range-button"),
                html.Button("Max", id="range-max", n_clicks=0, className="range-button"),
            ], className="range-buttons"),
        ], className="control"),

        html.Div([
            html.Label("Rolling window (trading days)"),
            dcc.Input(id="window", type="number", value=3, min=1, step=1),
            html.Label("Price metric", className="sub-label"),
            dcc.Dropdown(
                id="metric",
                options=[
                    {"label": "Closing Price", "value": "Close"},
                    {"label": "Adjusted Close", "value": "Adjusted Close"},
                ],
                value="Close", clearable=False
            ),
            html.Label("Benchmark return (%)", className="sub-label"),
            dcc.Input(id="benchmark", type="number", value=1.0, step=0.1),
            html.Label("Chart type", className="sub-label"),
            dcc.Dropdown(
                id="chart-type",
                options=[
                    {"label": "Line", "value": "line"},
                    {"label": "Candlestick (single stock)", "value": "candlestick"},
                    {"label": "Area", "value": "area"},
                    {"label": "OHLC (single stock)", "value": "ohlc"},
                ],
                value="line", clearable=False
            ),
        ], className="control"),
    ], className="controls"),

    html.Div([
        html.Label("Financial ratios"),
        dcc.Dropdown(
            id="ratio-selection",
            options=[{"label":v,"value":k} for k,v in RATIO_LABELS.items()],
            value=DEFAULT_RATIOS,
            multi=True,
            clearable=False
        ),
        html.Small(
            "Five key ratios are selected by default. Use the dropdown to add more.",
            style={"display":"block","opacity":".65","marginTop":"6px"}
        ),
    ], className="ratio-controls"),

    html.Div([
        html.Button("↻ Reset", id="reset", n_clicks=0, className="reset-button"),
        html.Button("✓ Analyse", id="analyse", n_clicks=0, className="analyse-button"),
    ], className="action-buttons"),

    dcc.Loading(
        id="analysis-loading",
        type="dot",
        children=html.Div(id="status", className="status")
    ),

    html.Div(id="cards", className="cards"),

    html.Div([
        html.Div(dcc.Graph(id="price-chart"), className="chart"),
        html.Div(dcc.Graph(id="return-chart"), className="chart"),
    ], className="chart-row"),

    html.Div([
        html.H3("Analysis Results"),
        dash_table.DataTable(
            id="results-table",
            columns=[],
            data=[],
            page_size=15,
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto"},
            style_cell={"padding": "8px", "textAlign": "left"},
            style_header={"fontWeight": "bold"},
        ),
        html.H3("Financial Ratios"),
        dash_table.DataTable(
            id="ratio-table",
            columns=[],
            data=[],
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={"padding": "8px", "textAlign": "left"},
            style_header={"fontWeight": "bold"},
        )
    ], className="table-panel"),

    html.Div(id="statistics-panel", className="statistics"),
], className="page")

@app.callback(
    Output("ticker", "value"),
    Output("custom-ticker", "value"),
    Output("mode", "value"),
    Output("window", "value"),
    Output("metric", "value"),
    Output("benchmark", "value"),
    Output("chart-type", "value"),
    Output("ratio-selection", "value"),
    Input("reset", "n_clicks"),
    prevent_initial_call=True,
)
def reset_dashboard(n_clicks):
    return ["GOLDBEES.NS"], "", "single", 3, "Close", 1.0, "line", DEFAULT_RATIOS


@app.callback(
    Output("date-range", "min_date_allowed"),
    Output("date-range", "max_date_allowed"),
    Output("date-range", "start_date"),
    Output("date-range", "end_date"),
    Output("status", "children"),
    Input("ticker", "value"),
    Input("custom-ticker", "value"),
    Input("reset", "n_clicks"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
)
def sync_date_range(tickers, custom, reset_clicks, current_start, current_end):
    selected = [x for x in (tickers or []) if x]
    if (custom or "").strip():
        selected.append(custom.strip())

    if not selected:
        return None, None, None, None, "Select at least one stock."

    try:
        starts, ends = [], []
        for ticker in selected:
            data = download_history(ticker)
            starts.append(pd.Timestamp(data.index.min()).normalize())
            ends.append(pd.Timestamp(data.index.max()).normalize())

        # The valid picker range is the common historical range of all selected stocks,
        # capped at today's date. It is never hard-coded.
        lo = max(starts)
        hi = min(min(ends), pd.Timestamp(date.today()).normalize())

        if lo > hi:
            raise ValueError("Selected stocks have no overlapping historical data.")

        # Explicit reset: latest one year.
        if reset_clicks:
            new_start = max(lo, hi - pd.Timedelta(days=365))
            new_end = hi
        else:
            # Preserve existing dates whenever they remain valid.
            old_start = pd.Timestamp(current_start).normalize() if current_start else None
            old_end = pd.Timestamp(current_end).normalize() if current_end else None

            if old_start is None or old_end is None:
                new_start = max(lo, hi - pd.Timedelta(days=365))
                new_end = hi
            else:
                new_start = max(lo, min(old_start, hi))
                new_end = max(lo, min(old_end, hi))

                # If the selected stocks make the old range invalid, use the
                # latest valid one-year range instead of producing an inverted range.
                if new_start > new_end:
                    new_start = max(lo, hi - pd.Timedelta(days=365))
                    new_end = hi

        label = ", ".join(selected[:4]) + ("..." if len(selected) > 4 else "")
        return (
            lo.date(),
            hi.date(),
            new_start.date(),
            new_end.date(),
            f"Available common history for {label}: {lo.date()} → {hi.date()}."
        )
    except Exception as e:
        return None, None, None, None, f"Could not load historical range: {e}"

@app.callback(
    Output("date-range", "start_date", allow_duplicate=True),
    Output("date-range", "end_date", allow_duplicate=True),
    Input("range-1m", "n_clicks"),
    Input("range-3m", "n_clicks"),
    Input("range-6m", "n_clicks"),
    Input("range-1y", "n_clicks"),
    Input("range-5y", "n_clicks"),
    Input("range-max", "n_clicks"),
    State("date-range", "min_date_allowed"),
    State("date-range", "max_date_allowed"),
    prevent_initial_call=True,
)
def apply_quick_range(n1, n3, n6, n1y, n5y, nmax, min_allowed, max_allowed):
    from dash import ctx

    if not max_allowed:
        return None, None

    hi = pd.Timestamp(max_allowed).normalize()
    lo = pd.Timestamp(min_allowed).normalize() if min_allowed else hi
    clicked = ctx.triggered_id

    spans = {
        "range-1m": 31,
        "range-3m": 92,
        "range-6m": 183,
        "range-1y": 365,
        "range-5y": 365 * 5,
    }

    if clicked == "range-max":
        start = lo
    else:
        start = max(lo, hi - pd.Timedelta(days=spans.get(clicked, 365)))

    return start.date(), hi.date()

@app.callback(
    Output("cards", "children"),
    Output("price-chart", "figure"),
    Output("return-chart", "figure"),
    Output("results-table", "columns"),
    Output("results-table", "data"),
    Output("statistics-panel", "children"),
    Output("status", "children", allow_duplicate=True),
    Output("ratio-table", "columns"),
    Output("ratio-table", "data"),
    Input("analyse", "n_clicks"),
    State("ticker", "value"),
    State("custom-ticker", "value"),
    State("mode", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    State("window", "value"),
    State("metric", "value"),
    State("benchmark", "value"),
    State("chart-type", "value"),
    State("ratio-selection", "value"),
    prevent_initial_call=True,
)
def analyse(n, ticker, custom, mode, start_date, end_date, window, metric, benchmark, chart_type, ratio_selection):
    selected = list(ticker or [])
    if (custom or "").strip():
        selected.append(custom.strip())
    selected = list(dict.fromkeys(selected))
    empty = go.Figure()

    try:
        if not selected:
            raise ValueError("Please select at least one stock.")
        if len(selected) > 10:
            raise ValueError("You can compare a maximum of 10 stocks.")
        window = max(1, int(window or 3))
        benchmark_pct = float(benchmark or 0)
        benchmark_fraction = benchmark_pct / 100

        histories = {}
        series = {}
        available_starts, available_ends = [], []
        for symbol in selected:
            data = download_history(symbol)
            s, used_col = clean_price_series(data, metric)
            histories[symbol] = (s, used_col)
            series[symbol] = s
            available_starts.append(s.index.min())
            available_ends.append(s.index.max())

        common_lo = max(available_starts)
        common_hi = min(available_ends)
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        if start < common_lo or end > common_hi:
            raise ValueError(
                f"Selected dates must fall within the common available history: "
                f"{common_lo.date()} to {common_hi.date()}."
            )
        if start > end:
            raise ValueError("Start date must be before end date.")

        rows_by_stock = {}
        comparison_rows = []

        for symbol, (s, used_col) in histories.items():
            filtered = s.loc[(s.index >= start) & (s.index <= end)]
            if filtered.empty:
                raise ValueError(f"No observations for {symbol} in the selected range.")

            start_value, start_trade = trading_value(s, start, window)
            end_value, end_trade = trading_value(s, end, window)
            total_return = (end_value / start_value - 1) * 100

            daily_returns = filtered.pct_change().dropna()
            volatility = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 1 else np.nan
            running_max = filtered.cummax()
            max_drawdown = (filtered / running_max - 1).min() * 100

            rows_by_stock[symbol] = {
                "series": filtered, "start_value": start_value, "end_value": end_value,
                "start_trade": start_trade, "end_trade": end_trade,
                "return": total_return, "volatility": volatility,
                "max_drawdown": max_drawdown, "used_col": used_col,
            }
            comparison_rows.append({
                "Stock": symbol,
                "Baseline Trading Day": str(start_trade.date()),
                "Comparison Trading Day": str(end_trade.date()),
                "Baseline Rolling Mean": round(start_value, 4),
                "Comparison Rolling Mean": round(end_value, 4),
                "Return (%)": round(total_return, 4),
                "Volatility (%)": round(volatility, 4) if not np.isnan(volatility) else None,
                "Max Drawdown (%)": round(max_drawdown, 4),
            })

        # Price chart: normalized comparison for multiple stocks; selected standard chart for one.
        chart = go.Figure()
        if len(selected) == 1:
            symbol = selected[0]
            info = rows_by_stock[symbol]
            s = info["series"]
            if chart_type == "candlestick":
                raw = histories[symbol][0].loc[s.index]
                chart.add_trace(go.Candlestick(
                    x=raw.index, open=histories[symbol][0].loc[raw.index],
                    high=histories[symbol][0].loc[raw.index],
                    low=histories[symbol][0].loc[raw.index],
                    close=raw.values, name=symbol
                ))
                # Replace the invalid OHLC construction above with the actual source columns when available.
                data = download_history(symbol)
                chart = go.Figure()
                chart.add_trace(go.Candlestick(
                    x=data.index, open=data["Open"], high=data["High"],
                    low=data["Low"], close=data["Close"], name=symbol
                ))
            elif chart_type == "ohlc":
                data = download_history(symbol)
                chart.add_trace(go.Ohlc(
                    x=data.index, open=data["Open"], high=data["High"],
                    low=data["Low"], close=data["Close"], name=symbol
                ))
            elif chart_type == "area":
                chart.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines",
                                           fill="tozeroy", name=symbol))
            else:
                chart.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=symbol))
            chart.update_layout(title=f"{symbol} — Price History",
                                xaxis_title="Date", yaxis_title="Price",
                                template="plotly_white", hovermode="x unified")
        else:
            for symbol, info in rows_by_stock.items():
                s = info["series"]
                normalized = s / info["start_value"] * 100
                chart.add_trace(go.Scatter(x=normalized.index, y=normalized.values,
                                           mode="lines", name=symbol))
            chart.update_layout(
                title="Relative Performance — Start = 100",
                xaxis_title="Date", yaxis_title="Indexed Value",
                template="plotly_white", hovermode="x unified"
            )

        # Return comparison.
        ret_chart = go.Figure()
        ret_chart.add_trace(go.Bar(
            x=[r["Stock"] for r in comparison_rows],
            y=[r["Return (%)"] for r in comparison_rows],
            name="Return"
        ))
        ret_chart.add_hline(y=benchmark_pct, line_dash="dash",
                            annotation_text=f"Benchmark {benchmark_pct:.2f}%")
        ret_chart.update_layout(title="Period Returns",
                                xaxis_title="Stock", yaxis_title="Return (%)",
                                template="plotly_white")

        # Statistics apply to exactly one selected stock.
        if len(selected) == 1:
            test_symbol = selected[0]
            info = rows_by_stock[test_symbol]
            test_returns = pd.Series([info["return"] / 100.0])
            stat_source = "single-period"
            # Historical repeated periods can generate a proper sample.
            if mode == "historical":
                s = histories[test_symbol][0]
                historical_rows = []
                for year in range(start.year, end.year + 1):
                    try:
                        b_date = pd.Timestamp(year=year, month=start.month, day=start.day)
                        c_date = pd.Timestamp(year=year, month=end.month, day=end.day)
                        if b_date < s.index.min() or c_date > s.index.max():
                            continue
                        b_val, b_trade = trading_value(s, b_date, window)
                        c_val, c_trade = trading_value(s, c_date, window)
                        r = (c_val / b_val - 1) * 100
                        historical_rows.append({
                            "Year": year,
                            "Baseline Trading Day": str(b_trade.date()),
                            "Comparison Trading Day": str(c_trade.date()),
                            "Baseline Rolling Mean": round(b_val, 4),
                            "Comparison Rolling Mean": round(c_val, 4),
                            "Return (%)": round(r, 4),
                        })
                    except Exception:
                        continue
                if historical_rows:
                    test_returns = pd.Series([r["Return (%)"] / 100 for r in historical_rows])
                    stat_source = "historical"
                    comparison_rows = historical_rows
                    ret_chart = go.Figure()
                    ret_chart.add_trace(go.Bar(
                        x=[r["Year"] for r in historical_rows],
                        y=[r["Return (%)"] for r in historical_rows],
                        name="Historical Return"
                    ))
                    ret_chart.add_hline(y=benchmark_pct, line_dash="dash",
                                        annotation_text=f"Benchmark {benchmark_pct:.2f}%")
                    ret_chart.update_layout(
                        title=f"Historical Returns: {start.strftime('%d-%b')} → {end.strftime('%d-%b')}",
                        xaxis_title="Year", yaxis_title="Return (%)", template="plotly_white"
                    )
        else:
            test_symbol = None
            test_returns = pd.Series(dtype=float)
            stat_source = "comparison"

        stats = descriptive(test_returns * 100)
        stat_tests = test_statistics(test_returns, benchmark_fraction) if stat_source == "historical" else None

        # Cards use first selected stock for the single-period overview when multiple are selected.
        primary = rows_by_stock[selected[0]]
        cards = [
            card("Stocks selected", str(len(selected))),
            card("Primary stock return", f"{primary['return']:+.2f}%"),
            card("Primary volatility", "N/A" if np.isnan(primary["volatility"]) else f"{primary['volatility']:.2f}%"),
            card("Primary max drawdown", f"{primary['max_drawdown']:.2f}%"),
            card("Benchmark", f"{benchmark_pct:.2f}%"),
            card("Test stock", test_symbol or "Select below"),
        ]

        stat_children = [html.H3("Statistical Analysis")]
        if len(selected) > 1:
            stat_children += [
                html.P("Multiple stocks are being compared. Statistical tests are run on one stock only."),
                dcc.Dropdown(
                    id="test-stock",
                    options=[{"label": x, "value": x} for x in selected],
                    value=selected[0], clearable=False
                ),
                html.P("V1 requires re-analysis after changing the test stock.", className="warning")
            ]
        elif stat_source == "historical" and len(test_returns) >= 5:
            stat_tests = test_statistics(test_returns, benchmark_fraction)
            stat_children += [
                html.P(f"Testing: {test_symbol} against a {benchmark_pct:.2f}% benchmark."),
                html.Div([
                    html.Span(f"Mean: {stats.get('Mean', np.nan):.2f}%"),
                    html.Span(f"Median: {stats.get('Median', np.nan):.2f}%"),
                    html.Span(f"Std Dev: {stats.get('Std Dev', np.nan):.2f}%"),
                    html.Span(f"Observations: {stats.get('Observations', 0)}"),
                ], className="stat-line"),
                html.Div([
                    html.Span(f"T-test p-value: {stat_tests['T-test p-value']:.4f}"),
                    html.Span(f"Wilcoxon p-value: {stat_tests['Wilcoxon p-value']:.4f}"),
                ], className="stat-line")
            ]
        else:
            stat_children += [
                html.P("Single-period mode has one return observation; hypothesis tests require historical repeated periods."),
                html.P("Use Historical repeated periods to create a sample for the t-test and Wilcoxon test.",
                        className="warning")
            ]

        columns = [{"name": c, "id": c} for c in comparison_rows[0].keys()] if comparison_rows else []
        status = (
            f"Analysed {len(selected)} stock(s). Common available history: "
            f"{common_lo.date()} to {common_hi.date()}."
        )
        ratio_rows = get_ratios(selected, ratio_selection or DEFAULT_RATIOS)
        ratio_columns = [{"name": c, "id": c} for c in ratio_rows[0].keys()] if ratio_rows else []
        return cards, chart, ret_chart, columns, comparison_rows, stat_children, status, ratio_columns, ratio_rows

    except Exception as e:
        return [], empty, empty, [], [], [
            html.H3("Statistical Analysis"),
            html.P("Analysis could not be completed.", className="warning")
        ], f"Error: {e}", [], []

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)

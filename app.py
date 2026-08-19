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
        "GoldBeES": "GOLDBEES.NS", "SBI Gold ETF": "SETFGOLD.NS",
        "Nippon India Silver ETF": "SILVERBEES.NS",
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

app = Dash(__name__, title="Stock Analytics Dashboard")
server = app.server

app.layout = html.Div([
    html.Div([
        html.H1("Stock Analytics Dashboard"),
        html.P("Flexible historical analysis of stocks, ETFs and indices using Yahoo Finance.")
    ], className="hero"),

    html.Div([
        html.Div([
            html.Label("Stock / ETF / Index"),
            dcc.Dropdown(
                id="ticker", options=options(), value="GOLDBEES.NS",
                searchable=True, clearable=False
            ),
            html.Label("Custom Yahoo ticker", className="sub-label"),
            dcc.Input(id="custom-ticker", type="text", placeholder="e.g. RELIANCE.NS",
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
            html.Label("Start date", className="sub-label"),
            dcc.DatePickerSingle(id="start-date", display_format="DD-MMM-YYYY"),
            html.Label("End / comparison date", className="sub-label"),
            dcc.DatePickerSingle(id="end-date", display_format="DD-MMM-YYYY"),
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
        ], className="control"),
    ], className="controls"),

    html.Button("Analyse", id="analyse", n_clicks=0, className="analyse-button"),
    html.Div(id="status", className="status"),

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
        )
    ], className="table-panel"),

    html.Div(id="statistics-panel", className="statistics"),
], className="page")

@app.callback(
    Output("start-date", "min_date_allowed"),
    Output("start-date", "max_date_allowed"),
    Output("start-date", "date"),
    Output("end-date", "min_date_allowed"),
    Output("end-date", "max_date_allowed"),
    Output("end-date", "date"),
    Output("status", "children"),
    Input("ticker", "value"),
    Input("custom-ticker", "value"),
)
def update_dates(ticker, custom):
    selected = (custom or "").strip() or ticker
    try:
        data = download_history(selected)
        lo, hi = data.index.min().date(), data.index.max().date()
        start = max(lo, hi - pd.Timedelta(days=365))
        return lo, hi, start, lo, hi, hi, (
            f"Available Yahoo Finance history for {selected}: {lo} to {hi}."
        )
    except Exception as e:
        return None, None, None, None, None, None, f"Could not load {selected}: {e}"

@app.callback(
    Output("cards", "children"),
    Output("price-chart", "figure"),
    Output("return-chart", "figure"),
    Output("results-table", "columns"),
    Output("results-table", "data"),
    Output("statistics-panel", "children"),
    Output("status", "children", allow_duplicate=True),
    Input("analyse", "n_clicks"),
    State("ticker", "value"),
    State("custom-ticker", "value"),
    State("mode", "value"),
    State("start-date", "date"),
    State("end-date", "date"),
    State("window", "value"),
    State("metric", "value"),
    State("benchmark", "value"),
    prevent_initial_call=True,
)
def analyse(n, ticker, custom, mode, start_date, end_date, window, metric, benchmark):
    selected = (custom or "").strip() or ticker
    empty = go.Figure()
    try:
        window = max(1, int(window or 3))
        benchmark = float(benchmark or 0) / 100
        data = download_history(selected)
        s, used_col = clean_price_series(data, metric)
        lo, hi = s.index.min().date(), s.index.max().date()

        if not start_date or not end_date:
            raise ValueError("Please select both dates.")
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        if start < s.index.min() or end > s.index.max():
            raise ValueError(f"Dates must fall within available history: {lo} to {hi}.")
        if start > end:
            raise ValueError("Start date must be before end date.")

        filtered = s.loc[(s.index >= start) & (s.index <= end)]
        if filtered.empty:
            raise ValueError("No trading observations exist in the selected range.")

        # Single-period analysis
        start_value, start_trade = trading_value(s, start, window)
        end_value, end_trade = trading_value(s, end, window)
        total_return = (end_value / start_value - 1) * 100
        daily_returns = filtered.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 1 else np.nan
        running_max = filtered.cummax()
        drawdown = (filtered / running_max - 1) * 100
        max_drawdown = drawdown.min()

        chart = go.Figure()
        chart.add_trace(go.Scatter(x=filtered.index, y=filtered.values,
                                   mode="lines", name=used_col))
        chart.update_layout(title=f"{selected} — Price History",
                            xaxis_title="Date", yaxis_title="Price",
                            template="plotly_white", hovermode="x unified")

        ret_chart = go.Figure()
        ret_chart.add_trace(go.Bar(
            x=[end_trade], y=[total_return],
            name="Period Return"
        ))
        ret_chart.add_hline(y=float(benchmark * 100), line_dash="dash",
                            annotation_text=f"Benchmark {benchmark*100:.2f}%")
        ret_chart.update_layout(title="Period Return",
                                xaxis_title="Comparison Date",
                                yaxis_title="Return (%)",
                                template="plotly_white")

        rows = [{
            "Date": str(end_trade.date()),
            "Baseline Trading Day": str(start_trade.date()),
            "Baseline Rolling Mean": round(start_value, 4),
            "Comparison Rolling Mean": round(end_value, 4),
            "Return (%)": round(total_return, 4),
        }]

        stats = descriptive(pd.Series([total_return]))
        stat_tests = None

        if mode == "historical":
            years = range(start.year, end.year + 1)
            historical_rows = []
            for year in years:
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

            rows = historical_rows
            returns = pd.Series([r["Return (%)"] / 100 for r in rows], dtype=float)
            stats = descriptive(returns * 100)
            stat_tests = test_statistics(returns, benchmark)

            ret_chart = go.Figure()
            ret_chart.add_trace(go.Bar(
                x=[r["Year"] for r in rows],
                y=[r["Return (%)"] for r in rows],
                name="Historical Return"
            ))
            ret_chart.add_hline(y=benchmark * 100, line_dash="dash",
                                annotation_text=f"Benchmark {benchmark*100:.2f}%")
            ret_chart.update_layout(
                title=f"Historical Returns: {start.strftime('%d-%b')} → {end.strftime('%d-%b')}",
                xaxis_title="Year", yaxis_title="Return (%)",
                template="plotly_white"
            )

        cards = [
            card("Baseline rolling mean", f"₹{start_value:,.2f}"),
            card("Comparison rolling mean", f"₹{end_value:,.2f}"),
            card("Return", f"{total_return:+.2f}%"),
            card("Annualised volatility", "N/A" if np.isnan(volatility) else f"{volatility:.2f}%"),
            card("Maximum drawdown", f"{max_drawdown:.2f}%"),
            card("Benchmark", f"{benchmark*100:.2f}%"),
        ]

        stat_children = [
            html.H3("Statistical Analysis"),
            html.P(
                "Statistical hypothesis tests are shown only in Historical repeated-period mode "
                "with at least 5 usable observations."
            ),
            html.Div([
                html.Span(f"Mean: {stats.get('Mean', np.nan):.2f}%"),
                html.Span(f"Median: {stats.get('Median', np.nan):.2f}%"),
                html.Span(f"Std Dev: {stats.get('Std Dev', np.nan):.2f}%"),
                html.Span(f"Observations: {stats.get('Observations', 0)}"),
            ], className="stat-line")
        ]

        if stat_tests:
            stat_children.append(html.Div([
                html.Span(f"T-test p-value: {stat_tests['T-test p-value']:.4f}"),
                html.Span(f"Wilcoxon p-value: {stat_tests['Wilcoxon p-value']:.4f}"),
            ], className="stat-line"))
        elif mode == "historical":
            stat_children.append(html.P(
                "Insufficient observations for reliable hypothesis testing (minimum 5).",
                className="warning"
            ))
        else:
            stat_children.append(html.P(
                "Single-period mode produces one return observation; hypothesis tests are disabled.",
                className="warning"
            ))

        columns = [{"name": c, "id": c} for c in rows[0].keys()] if rows else []
        status = f"Analysed {selected}. Available history: {lo} to {hi}. "
        status += f"Using {len(filtered)} trading observations in the selected period."

        return cards, chart, ret_chart, columns, rows, stat_children, status

    except Exception as e:
        return [], empty, empty, [], [], [
            html.H3("Statistical Analysis"),
            html.P("Analysis could not be completed.", className="warning")
        ], f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)

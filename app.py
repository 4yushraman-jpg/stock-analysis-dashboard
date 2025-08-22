import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# --- DATA LOADING ---
@st.cache_data
def load_data(path="processed_stock_data.pkl"):
    df = pd.read_pickle(path)

    # Ensure "date" is a datetime index
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Data must contain a 'date' column or a DatetimeIndex.")

    return df

st.title("📈 Stock Analysis Dashboard")
st.write("An interactive dashboard to analyze daily stock prices for major tech companies.")

# --- LOAD PROCESSED DATA ---
try:
    df = load_data("processed_stock_data.pkl")
    symbols = sorted(df["symbol"].unique())
except FileNotFoundError:
    st.error("❌ Data file not found. Please wait for the daily update or run `main.py` locally to generate it.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Error loading data: {e}")
    st.stop()

# --- SIDEBAR FOR USER INPUT ---
st.sidebar.header("User Controls")
selected_symbol = st.sidebar.selectbox("Select a Stock Symbol", symbols)

# Date range controls
min_date = df.index.min().date()
max_date = df.index.max().date()
start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

# --- FILTER DATA ---
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

date_filtered_df = df[(df.index >= start_date) & (df.index <= end_date)]
stock_df = date_filtered_df[date_filtered_df["symbol"] == selected_symbol].copy()

if stock_df.empty:
    st.warning("⚠️ No data available for this selection.")
    st.stop()

# --- KPIs ---
st.header(f"Analysis for: {selected_symbol}")

latest_price = stock_df["close"].iloc[-1]
period_start_price = stock_df["close"].iloc[0]
price_change = latest_price - period_start_price
price_change_pct = (price_change / period_start_price) * 100
highest_price = stock_df["close"].max()
lowest_price = stock_df["close"].min()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Latest Price", f"${latest_price:,.2f}")
with col2:
    st.metric("Change in Period", f"${price_change:,.2f}", f"{price_change_pct:.2f}%")
with col3:
    st.metric("Highest Price", f"${highest_price:,.2f}")
with col4:
    st.metric("Lowest Price", f"${lowest_price:,.2f}")

# --- CHARTS ---
st.subheader("Interactive Price Chart")
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=stock_df.index, y=stock_df["close"], mode="lines", name="Close Price"))

# Add moving averages if available
if "MA50" in stock_df.columns:
    fig_price.add_trace(go.Scatter(x=stock_df.index, y=stock_df["MA50"], mode="lines", name="50-Day MA", line=dict(dash="dot")))
if "MA200" in stock_df.columns:
    fig_price.add_trace(go.Scatter(x=stock_df.index, y=stock_df["MA200"], mode="lines", name="200-Day MA", line=dict(dash="dash")))

fig_price.update_layout(title_text=f"{selected_symbol} Closing Price", xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(fig_price, use_container_width=True)

# Layout for Volume + Risk vs Return
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Trading Volume")
    if "volume" in stock_df.columns:
        fig_vol = px.bar(stock_df, x=stock_df.index, y="volume", title=f"{selected_symbol} Trading Volume")
        fig_vol.update_layout(xaxis_title="Date", yaxis_title="Volume")
        st.plotly_chart(fig_vol, use_container_width=True)
    else:
        st.info("No volume data available.")

with col_right:
    st.subheader("Risk vs. Return Analysis (Full Period)")
    if "daily_return" in df.columns:
        returns_df = df.groupby("symbol")["daily_return"].agg(["mean", "std"]).rename(
            columns={"mean": "avg_return", "std": "risk"}
        ).dropna()
        returns_df["avg_return"] = returns_df["avg_return"] * 252
        returns_df["risk"] = returns_df["risk"] * (252**0.5)

        fig_risk = px.scatter(
            returns_df,
            x="risk",
            y="avg_return",
            text=returns_df.index,
            title="Annualized Risk vs. Return",
            labels={"risk": "Risk (Annualized Volatility)", "avg_return": "Average Return (Annualized)"}
        )
        fig_risk.update_traces(textposition="top center")

        if selected_symbol in returns_df.index:
            fig_risk.add_trace(
                go.Scatter(
                    x=[returns_df.loc[selected_symbol, "risk"]],
                    y=[returns_df.loc[selected_symbol, "avg_return"]],
                    mode="markers+text",
                    marker=dict(color="red", size=12),
                    text=[selected_symbol],
                    textposition="bottom center",
                    name="Selected"
                )
            )
        st.plotly_chart(fig_risk, use_container_width=True)
    else:
        st.info("No return data available.")

# --- RAW DATA ---
if st.sidebar.checkbox("Show Raw Data Table"):
    st.subheader("Raw Data for Selected Period and Stock")
    st.write(stock_df)

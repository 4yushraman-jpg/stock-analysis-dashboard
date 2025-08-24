import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DATA LOADING ---
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data(path="processed_stock_data.pkl"):
    try:
        df = pd.read_pickle(path)
        
        # Ensure "date" is a datetime index
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Data must contain a 'date' column or a DatetimeIndex.")
            
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please run the data processing script.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# --- HEADER ---
st.title("Stock Analysis Dashboard")
st.write("Interactive dashboard for analyzing daily stock prices of major tech companies.")

# --- LOAD PROCESSED DATA ---
df = load_data("processed_stock_data.pkl")
symbols = sorted(df["symbol"].unique())

# --- SIDEBAR CONTROLS ---
st.sidebar.header("User Controls")
selected_symbol = st.sidebar.selectbox("Select Stock Symbol", symbols)

# Date range controls with sensible defaults
min_date, max_date = df.index.min().date(), df.index.max().date()
default_end = max_date
default_start = max_date - timedelta(days=365)  # Default to 1 year

if default_start < min_date:
    default_start = min_date

start_date = st.sidebar.date_input("Start Date", default_start, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", default_end, min_value=min_date, max_value=max_date)

# Additional analysis options
st.sidebar.subheader("Analysis Options")
show_volume = st.sidebar.checkbox("Show Volume Chart", value=True)
show_technical_indicators = st.sidebar.checkbox("Show Technical Indicators", value=True)
benchmark_symbol = st.sidebar.selectbox("Compare with Benchmark", ["None"] + [s for s in symbols if s != selected_symbol])

# --- FILTER DATA ---
start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
date_filtered_df = df[(df.index >= start_date) & (df.index <= end_date)]
stock_df = date_filtered_df[date_filtered_df["symbol"] == selected_symbol].copy()

if stock_df.empty:
    st.warning("No data available for this selection.")
    st.stop()

# --- KPI METRICS ---
st.header(f"Analysis for: {selected_symbol}")

latest_price = stock_df["close"].iloc[-1]
period_start_price = stock_df["close"].iloc[0]
price_change = latest_price - period_start_price
price_change_pct = (price_change / period_start_price) * 100
highest_price, lowest_price = stock_df["close"].max(), stock_df["close"].min()
avg_volume = stock_df["volume"].mean() if "volume" in stock_df.columns else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest Price", f"${latest_price:,.2f}", f"{price_change_pct:.2f}%")
col2.metric("Period High", f"${highest_price:,.2f}")
col3.metric("Period Low", f"${lowest_price:,.2f}")
col4.metric("Avg Volume", f"{avg_volume:,.0f}" if avg_volume > 0 else "N/A")

# --- PRICE CHART ---
st.subheader("Price Chart")
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(
    x=stock_df.index, 
    y=stock_df["close"], 
    mode="lines", 
    name="Close Price",
    line=dict(width=2)
))

# Add technical indicators if selected and available
if show_technical_indicators:
    if "MA50" in stock_df.columns:
        fig_price.add_trace(go.Scatter(
            x=stock_df.index, 
            y=stock_df["MA50"], 
            mode="lines", 
            name="50-Day MA", 
            line=dict(dash="dot", width=1.5)
        ))
    if "MA200" in stock_df.columns:
        fig_price.add_trace(go.Scatter(
            x=stock_df.index, 
            y=stock_df["MA200"], 
            mode="lines", 
            name="200-Day MA", 
            line=dict(dash="dash", width=1.5)
        ))

# Add benchmark comparison if selected
if benchmark_symbol != "None":
    benchmark_df = date_filtered_df[date_filtered_df["symbol"] == benchmark_symbol].copy()
    if not benchmark_df.empty:
        # Normalize prices for comparison
        norm_stock = stock_df["close"] / stock_df["close"].iloc[0]
        norm_benchmark = benchmark_df["close"] / benchmark_df["close"].iloc[0]
        
        fig_price.add_trace(go.Scatter(
            x=benchmark_df.index, 
            y=norm_benchmark * 100, 
            mode="lines", 
            name=f"{benchmark_symbol} (Normalized)",
            line=dict(width=1.5, color="purple"),
            yaxis="y2"
        ))
        
        fig_price.update_layout(
            yaxis2=dict(
                title="Benchmark (%)",
                overlaying="y",
                side="right",
                showgrid=False
            )
        )

fig_price.update_layout(
    title=f"{selected_symbol} Closing Price",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_price, use_container_width=True)

# --- VOLUME + RISK/RETURN ---
col_left, col_right = st.columns(2)

with col_left:
    if show_volume and "volume" in stock_df.columns:
        st.subheader("Trading Volume")
        fig_vol = px.bar(stock_df, x=stock_df.index, y="volume", title=f"{selected_symbol} Volume")
        fig_vol.update_layout(xaxis_title="Date", yaxis_title="Volume")
        st.plotly_chart(fig_vol, use_container_width=True)

with col_right:
    st.subheader("Risk vs Return (All Stocks)")
    if "daily_return" in df.columns:
        returns_df = df.groupby("symbol")["daily_return"].agg(["mean", "std"]).dropna()
        returns_df.rename(columns={"mean": "avg_return", "std": "risk"}, inplace=True)
        returns_df["avg_return"] *= 252  # Annualize
        returns_df["risk"] *= (252**0.5)  # Annualize
        
        # Calculate Sharpe ratio (assuming risk-free rate = 0 for simplicity)
        returns_df["sharpe"] = returns_df["avg_return"] / returns_df["risk"]
        
        # Create size array for highlighting selected symbol
        sizes = [15 if symbol == selected_symbol else 8 for symbol in returns_df.index]
        
        fig_risk = px.scatter(
            returns_df, x="risk", y="avg_return", 
            hover_data=["sharpe"],
            title="Annualized Risk vs Return",
            labels={"risk": "Risk (Volatility)", "avg_return": "Annualized Return"}
        )
        
        # Update marker sizes
        fig_risk.update_traces(marker=dict(size=sizes))
        
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # Show performance metrics for selected stock with explanation
        if selected_symbol in returns_df.index:
            sharpe_value = returns_df.loc[selected_symbol, 'sharpe']
            st.metric("Sharpe Ratio", f"{sharpe_value:.2f}")
            
            # Sharpe ratio explanation
            with st.expander("What is the Sharpe Ratio?"):
                st.write("""
                The Sharpe Ratio measures risk-adjusted return. It represents the average return 
                earned in excess of the risk-free rate per unit of volatility or total risk.
                
                - **Higher values** indicate better risk-adjusted performance
                - **Values above 1** are generally considered good
                - **Values above 2** are considered very good
                - **Values above 3** are considered excellent
                
                In this calculation, we've assumed a risk-free rate of 0 for simplicity.
                """)

# --- ADDITIONAL ANALYSIS ---
st.subheader("Additional Analysis")

# Daily returns distribution
if "daily_return" in stock_df.columns:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Daily Returns Distribution")
        fig_returns = px.histogram(
            stock_df, x="daily_return", 
            nbins=50, 
            title=f"{selected_symbol} Daily Returns Distribution"
        )
        st.plotly_chart(fig_returns, use_container_width=True)
    
    with col2:
        st.write("Cumulative Returns")
        cumulative_returns = (1 + stock_df["daily_return"]).cumprod() - 1
        fig_cumulative = px.line(
            x=stock_df.index, y=cumulative_returns * 100,
            title=f"{selected_symbol} Cumulative Returns (%)"
        )
        st.plotly_chart(fig_cumulative, use_container_width=True)

# --- RAW DATA + DOWNLOAD ---
st.subheader("Raw Data")
if st.sidebar.checkbox("Show Raw Data Table"):
    st.dataframe(stock_df, use_container_width=True)

    # Option to download CSV
    csv = stock_df.to_csv().encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{selected_symbol}_data.csv",
        mime="text/csv"
    )

# --- FOOTER ---
st.markdown("---")
st.caption(f"Data from {min_date} to {max_date}. Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
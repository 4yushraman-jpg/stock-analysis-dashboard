import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# --- DATA LOADING ---
@st.cache_data  # This decorator caches the data loading, so it doesn't re-run on every interaction
def load_data(path):
    df = pd.read_pickle(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.set_index('date', inplace=True)
    return df

st.title('📈 Stock Analysis Dashboard')
st.write("An interactive dashboard to analyze daily stock prices for major tech companies.")

# Load the processed data
try:
    df = load_data('processed_stock_data.pkl')
    symbols = sorted(df['symbol'].unique())
except FileNotFoundError:
    st.error("Error: 'processed_stock_data.pkl' not found. Please run your feature engineering notebook first.")
    st.stop()


# --- SIDEBAR FOR USER INPUT ---
st.sidebar.header('User Controls')
selected_symbol = st.sidebar.selectbox('Select a Stock Symbol', symbols)

# Date Range Selector
min_date = df.index.min().date()
max_date = df.index.max().date()
start_date = st.sidebar.date_input('Start Date', min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input('End Date', max_date, min_value=min_date, max_value=max_date)


# --- FILTER DATA BASED ON SELECTION ---
# Convert start_date and end_date to datetime objects for comparison
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Filter the main dataframe by date range
date_filtered_df = df[(df.index >= start_date) & (df.index <= end_date)]

# Filter for the selected stock
stock_df = date_filtered_df[date_filtered_df['symbol'] == selected_symbol].copy()


# --- DISPLAY KPIs (Key Performance Indicators) ---
st.header(f'Analysis for: {selected_symbol}')

# Calculate KPIs
latest_price = stock_df['close'].iloc[-1]
period_start_price = stock_df['close'].iloc[0]
price_change = latest_price - period_start_price
price_change_pct = (price_change / period_start_price) * 100
highest_price = stock_df['close'].max()
lowest_price = stock_df['close'].min()

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

# 1. Interactive Price Chart with Plotly
st.subheader('Interactive Price Chart')
fig_price = go.Figure()

# Add traces for Close Price and Moving Averages
fig_price.add_trace(go.Scatter(x=stock_df.index, y=stock_df['close'], mode='lines', name='Close Price'))
fig_price.add_trace(go.Scatter(x=stock_df.index, y=stock_df['MA50'], mode='lines', name='50-Day MA', line=dict(dash='dot')))
fig_price.add_trace(go.Scatter(x=stock_df.index, y=stock_df['MA200'], mode='lines', name='200-Day MA', line=dict(dash='dash')))

fig_price.update_layout(title_text=f'{selected_symbol} Closing Price', xaxis_title='Date', yaxis_title='Price (USD)')
st.plotly_chart(fig_price, use_container_width=True)


# Layout for the next row of charts
col_left, col_right = st.columns(2)

with col_left:
    # 2. Volume Chart with Plotly
    st.subheader('Trading Volume')
    fig_vol = px.bar(stock_df, x=stock_df.index, y='volume', title=f'{selected_symbol} Trading Volume')
    fig_vol.update_layout(xaxis_title='Date', yaxis_title='Volume')
    st.plotly_chart(fig_vol, use_container_width=True)

with col_right:
    # 3. Risk vs. Return Scatter Plot
    st.subheader('Risk vs. Return Analysis (Full Period)')
    
    # Calculate returns for all stocks in the full (unfiltered) dataframe
    returns_df = df.groupby('symbol')['daily_return'].agg(['mean', 'std']).rename(columns={'mean': 'avg_return', 'std': 'risk'}).dropna()
    returns_df['avg_return'] = returns_df['avg_return'] * 252
    returns_df['risk'] = returns_df['risk'] * (252**0.5)

    # Create the plot
    fig_risk = px.scatter(
        returns_df, 
        x='risk', 
        y='avg_return', 
        text=returns_df.index,
        title='Annualized Risk vs. Return',
        labels={'risk': 'Risk (Annualized Volatility)', 'avg_return': 'Average Return (Annualized)'}
    )
    # Highlight the selected stock
    fig_risk.update_traces(textposition='top center')  # <-- FIXED
    fig_risk.add_trace(
        go.Scatter(
            x=[returns_df.loc[selected_symbol, 'risk']],
            y=[returns_df.loc[selected_symbol, 'avg_return']],
            mode='markers+text',
            marker=dict(color='red', size=12),
            text=[selected_symbol],
            textposition='bottom center',  # <-- FIXED
            name='Selected'
        )
    )
    st.plotly_chart(fig_risk, use_container_width=True)

# --- DISPLAY RAW DATA TABLE ---
if st.sidebar.checkbox('Show Raw Data Table'):
    st.subheader('Raw Data for Selected Period and Stock')
    st.write(stock_df)

import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

# --- 1. CONFIGURATION ---
load_dotenv()  # Load .env locally; in GitHub Actions it will use secrets
API_KEY = os.getenv('API_KEY')
SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
OUTPUT_PICKLE_PATH = 'processed_stock_data.pkl'  # Final file for Streamlit app


# --- 2. EXTRACTION ---
def extract_data(symbols, api_key):
    """Fetches daily time series data for a list of symbols from Alpha Vantage."""
    all_data = []
    print("🚀 Starting Data Extraction...")
    for symbol in symbols:
        print(f"   → Fetching {symbol}...")
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=compact'
        try:
            r = requests.get(url)
            r.raise_for_status()
            data = r.json()

            if "Time Series (Daily)" not in data:
                print(f"⚠️  API Error/Limit for {symbol}: {data.get('Note', 'No data returned')}")
                continue

            time_series = data['Time Series (Daily)']
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df['symbol'] = symbol
            all_data.append(df)

            time.sleep(12)  # Respect API rate limits (5 calls/min)

        except requests.exceptions.RequestException as e:
            print(f"❌ HTTP Request failed for {symbol}: {e}")
        except Exception as e:
            print(f"❌ Unexpected error for {symbol}: {e}")

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data)


# --- 3. TRANSFORMATION ---
def transform_data(df):
    """Cleans and transforms the raw stock data."""
    if df.empty:
        return df

    print("🔄 Starting Data Transformation...")
    df.index = pd.to_datetime(df.index)
    df.reset_index(inplace=True)

    rename_map = {
        'index': 'date', '1. open': 'open', '2. high': 'high',
        '3. low': 'low', '4. close': 'close', '5. volume': 'volume'
    }
    df.rename(columns=rename_map, inplace=True)

    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# --- 4. FEATURE ENGINEERING ---
def feature_engineer(df):
    """Adds key financial metrics like moving averages and returns."""
    if df.empty:
        return df

    print("📈 Starting Feature Engineering...")
    df.sort_values(by=['symbol', 'date'], inplace=True)

    df['daily_return'] = df.groupby('symbol')['close'].pct_change()
    df['MA50'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=50).mean())
    df['MA200'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=200).mean())
    df['volatility'] = df.groupby('symbol')['daily_return'].transform(lambda x: x.rolling(window=30).std())

    return df


# --- 5. MAIN PIPELINE ---
def main():
    if not API_KEY:
        raise ValueError("❌ API_KEY not set! Check .env (local) or GitHub secrets (Actions).")

    raw_df = extract_data(SYMBOLS, API_KEY)

    if not raw_df.empty:
        transformed_df = transform_data(raw_df)
        final_df = feature_engineer(transformed_df)

        # Save final processed data
        final_df.to_pickle(OUTPUT_PICKLE_PATH)
        print(f"✅ Successfully processed {len(final_df)} rows "
              f"for {final_df['symbol'].nunique()} symbols.")
        print(f"📂 Data saved to: {OUTPUT_PICKLE_PATH}")
    else:
        print("⚠️ Pipeline finished: No data fetched from API.")


if __name__ == "__main__":
    main()

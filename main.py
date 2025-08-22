import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

# --- 1. CONFIGURATION ---
# For local runs, it will use the .env file. For GitHub Actions, it will use repository secrets.
load_dotenv()
API_KEY = os.getenv('API_KEY') 
SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
OUTPUT_PICKLE_PATH = 'processed_stock_data.pkl' # The single output file for our Streamlit app

def extract_data(symbols, api_key):
    """Fetches daily time series data for a list of symbols from Alpha Vantage."""
    all_data = []
    print("--- Starting Data Extraction ---")
    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=compact'
        try:
            r = requests.get(url)
            r.raise_for_status()
            data = r.json()
            
            # The API returns a "Note" key when the call limit is reached.
            if "Time Series (Daily)" not in data:
                print(f"API Error or note for {symbol}: {data.get('Note', 'No data returned')}")
                continue
            
            time_series = data['Time Series (Daily)']
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df['symbol'] = symbol
            all_data.append(df)
            time.sleep(12) # Respect API rate limits (5 calls/min)
        except requests.exceptions.RequestException as e:
            print(f"HTTP Request failed for {symbol}: {e}")
        except KeyError:
            print(f"Could not find 'Time Series (Daily)' key for {symbol}. Response: {data}")
        except Exception as e:
            print(f"An unexpected error occurred for {symbol}: {e}")
            
    if not all_data:
        return pd.DataFrame()
        
    return pd.concat(all_data)

def transform_data(df):
    """Cleans and transforms the raw stock data."""
    if df.empty:
        return df
        
    print("--- Starting Data Transformation ---")
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

def feature_engineer(df):
    """Adds key financial metrics like moving averages and returns."""
    if df.empty:
        return df
        
    print("--- Starting Feature Engineering ---")
    # Sort values to ensure correct rolling calculations
    df.sort_values(by=['symbol', 'date'], inplace=True)
    
    df['daily_return'] = df.groupby('symbol')['close'].pct_change()
    df['MA50'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=50).mean())
    df['MA200'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=200).mean())
    df['volatility'] = df.groupby('symbol')['daily_return'].transform(lambda x: x.rolling(window=30).std())
    
    return df

def main():
    """Main ETL and processing pipeline to generate the final data file."""
    if not API_KEY:
        raise ValueError("API_KEY environment variable not set! Please check your .env file or GitHub secrets.")
        
    raw_df = extract_data(SYMBOLS, API_KEY)
    
    if not raw_df.empty:
        transformed_df = transform_data(raw_df)
        final_df = feature_engineer(transformed_df)
        
        # Save the final processed data, overwriting the old file
        final_df.to_pickle(OUTPUT_PICKLE_PATH)
        print(f"Successfully processed and saved data for {len(final_df)} rows to '{OUTPUT_PICKLE_PATH}'")
    else:
        print("Pipeline finished: No data was fetched from the API.")

if __name__ == "__main__":
    main()
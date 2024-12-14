import requests
from datetime import datetime, timedelta
import csv
from collections import defaultdict
import sys  # For exiting the code

def fetch_crypto_data():
    """
    Fetching OHLC data for the specified cryptocurrency and currency from the CoinGecko API.
    Aggregates 4-hour intervals into daily high and low values.
    """
    # Getting start date input and converting to datetime
    print("Enter start date (YYYY-MM-DD):")
    start_date_input = input()
    start_date = datetime.strptime(start_date_input, "%Y-%m-%d")

    # Calculate the difference in days between the start date and today's date
    today = datetime.today()
    day_difference = (today - start_date).days

    # Check if the difference is more than 30 days
    if day_difference >= 30:
        print("Data Limit Exceeded: Only last 30 Days data will be available")

    # Getting cryptocurrency item name and currency
    print("Enter name of Crypto:")
    item = input()                                # Example: bitcoin
    print("Enter name of Currency:")
    curr = input()                                # Example: usd

    # Getting look-back and look-forward periods
    print("Enter days look-back period:")
    variable1 = int(input())
    print("Enter days look-forward period:")
    variable2 = int(input())
    if(variable1+variable2>=day_difference):
        print("The sum of look-back and look-forward period should be less than the number of days requested.")
        sys.exit()


    # Fetching last 30 days of OHLC data from CoinGecko API
    ohlc_data = requests.get(
        f"https://api.coingecko.com/api/v3/coins/{item}/ohlc",
        params={"vs_currency": curr, "days": 30}
    ).json()

    # Checking if data retrieval was successful
    if not ohlc_data:
        print("Error: Unable to fetch sufficient data.")
        return None, None, None, None, None

    # Processing and storing daily OHLC data with aggregated high and low values
    daily_ohlc = defaultdict(lambda: {'high': float('-inf'), 'low': float('inf'), 'open': None, 'close': None})

    for entry in ohlc_data:
        # Converting timestamp to date and extracting OHLC values for the 4-hour interval
        timestamp = entry[0] / 1000
        date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
        open_price, high_price, low_price, close_price = entry[1], entry[2], entry[3], entry[4]

        # Initializing open price for the first interval of the day
        if daily_ohlc[date]['open'] is None:
            daily_ohlc[date]['open'] = open_price
        
        # Setting close price for the last interval of the day
        daily_ohlc[date]['close'] = close_price

        # Updating daily high and low based on the interval's high and low
        daily_ohlc[date]['high'] = max(daily_ohlc[date]['high'], high_price)
        daily_ohlc[date]['low'] = min(daily_ohlc[date]['low'], low_price)

    # Converting default dictionary to a regular dictionary and returning values
    daily_ohlc = dict(daily_ohlc)
    return daily_ohlc, start_date, variable1, variable2

def calculate_metrics(daily_ohlc, start_date, variable1, variable2):
    """
    Calculate historical and future metrics for each date based on the OHLC data.
    Returns a filtered dictionary with calculated metrics from start_date onward.
    """
    # Sorting dates in ascending order for consistent calculation
    dates = sorted(daily_ohlc.keys())
    all_calculations = {}

    # Iterating over each date to calculate metrics
    for i in range(len(dates)):
        current_date = dates[i]
        close_price = daily_ohlc[current_date]['close']
        metrics = {}

        # Calculating historical metrics based on look-back period (variable1)
        if i >= variable1:
            # Getting data for the past 'variable1' days
            window_data = [daily_ohlc[dates[j]] for j in range(i - variable1, i)]
            high_d = max(day['high'] for day in window_data)  # Max high in the window
            low_d = min(day['low'] for day in window_data)    # Min low in the window
            
            # Determining the position of max/min in look-back window
            high_index = next(j for j, day in enumerate(window_data) if day['high'] == high_d)
            low_index = next(j for j, day in enumerate(window_data) if day['low'] == low_d)
            
            # Calculating days since the high/low and percentage difference from high/low
            metrics['high_d'] = high_d
            metrics['low_d'] = low_d
            metrics['days_since_high'] = variable1 - high_index
            metrics['days_since_low'] = variable1 - low_index
            metrics['percent_diff_from_high'] = ((close_price - high_d) / high_d) * 100
            metrics['percent_diff_from_low'] = ((close_price - low_d) / low_d) * 100
        else:
            # If insufficient data, set metrics to None
            metrics['high_d'] = None
            metrics['low_d'] = None
            metrics['days_since_high'] = None
            metrics['days_since_low'] = None
            metrics['percent_diff_from_high'] = None
            metrics['percent_diff_from_low'] = None

        # Calculating future metrics based on look-forward period (variable2)
        if i + variable2 < len(dates):
            # Getting data for the next 'variable2' days
            future_window_high = [daily_ohlc[dates[j]]['high'] for j in range(i + 1, i + 1 + variable2)]
            future_window_low = [daily_ohlc[dates[j]]['low'] for j in range(i + 1, i + 1 + variable2)]
            
            # Calculating future high/low and percentage difference from current close price
            metrics['future_high'] = max(future_window_high)
            metrics['future_low'] = min(future_window_low)
            metrics['percent_diff_from_high_future'] = ((close_price - metrics['future_high']) / metrics['future_high']) * 100
            metrics['percent_diff_from_low_future'] = ((close_price - metrics['future_low']) / metrics['future_low']) * 100
        else:
            metrics['future_high'] = None
            metrics['future_low'] = None
            metrics['percent_diff_from_high_future'] = None
            metrics['percent_diff_from_low_future'] = None

        # Storing calculated metrics in the dictionary
        all_calculations[current_date] = metrics

    # Filtering calculations to only include dates from start_date onward
    filtered_calculations = {date: all_calculations[date] for date in dates if datetime.strptime(date, "%Y-%m-%d") >= start_date}
    return filtered_calculations

def export_to_csv(filtered_calculations, daily_ohlc):
    """
    Export the filtered calculations and OHLC data to a CSV file.
    """
    # Preparing data rows for CSV export
    csv_data = [
        [
            date,
            daily_ohlc[date]['open'],
            daily_ohlc[date]['high'],
            daily_ohlc[date]['low'],
            daily_ohlc[date]['close'],
            metrics['high_d'],
            metrics['days_since_high'],
            metrics['percent_diff_from_high'],
            metrics['low_d'],
            metrics['days_since_low'],
            metrics['percent_diff_from_low'],
            metrics['future_high'],
            metrics['future_low'],
            metrics['percent_diff_from_high_future'],
            metrics['percent_diff_from_low_future']
        ]
        for date, metrics in filtered_calculations.items()
    ]

    # Define CSV header
    header = [
        "Date", "Open", "High", "Low", "Close", "Historical High Price", "Days_Since_High", "%_Diff_High",
        "Historical Low Price", "Days_Since_Low", "%_Diff_Low", "Future_High", "Future_Low",
        "% Difference from Future High", "% Difference from Future Low"
    ]

    # Write data to CSV file
    with open("output.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(csv_data)

    print("Data exported to output.csv successfully.")

# Main function to coordinate data fetching, metric calculation, and CSV export
if __name__ == "__main__":
    # Fetch cryptocurrency data from CoinGecko API
    daily_ohlc, start_date, variable1, variable2 = fetch_crypto_data()
    
    if daily_ohlc:
        # Calculate metrics for each date
        filtered_calculations = calculate_metrics(daily_ohlc, start_date, variable1, variable2)

        # Export results to a CSV file
        export_to_csv(filtered_calculations, daily_ohlc)
Data Collection: A function fetches historical OHLC (Open, High, Low, Close) data for a specified cryptocurrency and currency over the past 30 days, at 4-hour intervals (4 candels).


Data Aggregation and Metrics Calculation: Each day’s OHLC data is aggregated from 4-hour intervals into daily high, low, open, and close prices. Metrics like the historical high and low over a user-defined period, percentage differences, and days since the last high and low are calculated. Future metrics are also computed to predict trends.


Machine Learning Model: A model is trained to predict future price differences based on historical metrics. This model uses calculated features such as days since the last high or low, and percentage differences from historical highs and lows to make its predictions. The accuracy of the model is about 98%.

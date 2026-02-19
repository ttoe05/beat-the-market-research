# Data collection script ( fetching data from Yahoo Finance, Alpha Vantage, etc. and
# saving it to CSV files)

# from src.data.data_collector import DataCollector
# # Example usage
# if __name__ == "__main__":
#     import logging
#     from src.utils.logger import setup_root_logger
#
#     # Setup colored logging for the entire application
#     setup_root_logger(level=logging.INFO, log_file='logs/data_collector.log')
#
#     # Initialize collector
#     collector = DataCollector(source='yahoo')
#
#     # Fetch data
#     data = collector.fetch_data(
#         symbol='AAPL',
#         start='2020-01-01',
#         end='2024-12-31'
#     )
#
#     # Save data
#     collector.save_data(data, 'AAPL')
#
#     print(f"\nCollected {len(data)} rows of data")
#     print(data.head())


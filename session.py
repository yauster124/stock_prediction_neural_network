from collections import deque
from db_handler import DbHandler


class Session:
    """Stores all of the current session variables"""
    def __init__(self):
        self.current_user = None
        self.current_stock = None  # Stores the current stock that the user is watching.
        self.stock_dict = {}  # {stock_id: stock_ticker}
        self.current_df = None  # Stores the current dataframe. Changes if the user changes the stock.
        self.stock_change = True  # Checks if the user changes the current stock.
        self.time_change = False  # Checks if the user changes the timeframe.
        self.timeperiod = '1M'  # Changes the graph
        self.dbh = DbHandler()  # Initialises a database handler

    def update(self):
        self.stock_dict = self.dbh.get_user_stocks(self)

    def current_stock_ticker(self):
        """Returns the current stock ticker"""
        try:
            return self.stock_dict[self.current_stock]
        except AttributeError:
            return None
            


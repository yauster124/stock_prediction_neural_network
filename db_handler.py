import sqlite3
import yfinance as yf
from user import User


class DbHandler:
    """Utility class; handles database queries.
       _s denotes an SQL statement."""

    def __init__(self):
        # Initialise connection.
        self.conn = sqlite3.connect('StockForecaster.db')
        self.c = self.conn.cursor()
        
    def check_login(self, username, password, session):
        is_valid = False

        # Gets user's password.
        user_pass_s = f"SELECT Password FROM Users WHERE Username = '{username}'"  # Gets the user's password from the db.
        self.c.execute(user_pass_s)
        db_password = self.c.fetchone()

        # If the user's password exists.
        if db_password:
            # If the entered password is correct.
            if db_password[0] == password:
                is_valid = True

                # Gets user's id.
                user_id_s = f'SELECT person_id FROM Users WHERE Username = "{username}"'
                self.c.execute(user_id_s)
                user_id = self.c.fetchone()[0]

                # Initialises the current user
                session.current_user = User(user_id, username)
                session.stock_dict = self.get_user_stocks(session)

                # Sets the current stock.
                if session.stock_dict:
                    session.current_stock = next(iter(session.stock_dict))
                    
        self.conn.commit()

        return is_valid, session

    def add_user_to_db(self, record):
        is_valid = False
        self.c.execute(f'SELECT 1 FROM Users WHERE username = "{record[0]}"')
        if not self.c.fetchone():
            record = ', '.join([f'"{x}"' for x in record])
            statement = f"INSERT INTO Users (Username, Password) VALUES({record})"
            self.c.execute(statement)
            self.conn.commit()
            is_valid = True

        return is_valid

    def add_stock_to_db(self, stock_ticker, user_id):
        valid = False
        self.c.execute(f'SELECT stock_id FROM Stocks WHERE stock_ticker = "{stock_ticker}"')
        stock_exists = self.c.fetchone()

        # If the stock already exists in the Stocks table.
        if stock_exists:
            self.c.execute(f'SELECT stock_id FROM UsersStocks WHERE user_id = {user_id} AND stock_id = "{stock_exists[0]}"')
            user_stock_exists = self.c.fetchone()
            if not user_stock_exists:  # Checks if the user is already watching the stock
                self.c.execute(f'INSERT INTO UsersStocks (user_id, stock_id) VALUES ({user_id}, {stock_exists[0]})')
                valid = True
                
        else:
            stock_data = yf.download(stock_ticker)
            if len(stock_data.values) > 0:  # Checks if it is a valid stock
                self.c.execute(f'INSERT INTO Stocks (stock_ticker) VALUES ("{stock_ticker}")')
                self.c.execute(f'SELECT stock_id FROM Stocks WHERE stock_ticker = "{stock_ticker}"')
                stock_id = self.c.fetchone()[0]
                self.c.execute(f'INSERT INTO UsersStocks (user_id, stock_id) VALUES ({user_id}, {stock_id})')
                valid = True
        self.conn.commit()

        return valid

    def stocks_from_db(self, stock_id):
        stock_list = []
        for x in stock_id:
            self.c.execute(f'SELECT stock_ticker FROM Stocks WHERE stock_id = {x}')
            stock_list.append(self.c.fetchone()[0])
        self.conn.commit()

        return stock_list

    def get_stock_id(self, stock):
        self.c.execute(f'SELECT stock_id FROM Stocks WHERE stock_ticker = "{stock}"')
        stock_id = [self.c.fetchone()[0]]
        self.conn.commit()

        return stock_id

    def delete_stock(self, stock_id, user_id):
        """Deletes the user-stock relation from the UsersStocks table."""
        delete_user_stock_s = f'DELETE FROM UsersStocks WHERE stock_id = {stock_id} AND user_id = {user_id}'
        self.c.execute(delete_user_stock_s)
        self.conn.commit()

    def get_user_stocks(self, session):
        """Returns the user's stocks."""
        # Gets the stock ids associated with the user.
        user_stock_id_s = f'SELECT stock_id FROM UsersStocks WHERE user_id = {session.current_user.user_id}' 
        self.c.execute(user_stock_id_s)
        stock_ids = self.c.fetchall()

        # Initialise stock dict.
        session.stock_dict = {}
        for stock_id in stock_ids:
            # Gets stock ticker using the stock id.
            stock_ticker_s = f'SELECT stock_ticker FROM Stocks WHERE stock_id = {stock_id[0]}'
            self.c.execute(stock_ticker_s)
            stock_ticker = self.c.fetchone()[0]
            session.stock_dict[stock_id[0]] = stock_ticker
                
        return session.stock_dict



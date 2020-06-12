from keras.models import Sequential
from keras.layers import Dense
from keras.layers import CuDNNLSTM as cLSTM
from keras.layers import Dropout
from model_handler import ModelHandler
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import yfinance as yf


class LSTMModel:

    def __init__(self, epochs=None, batch_size=None, timestep=None, save=True):
        self.save = save
        self.myModelHandler = ModelHandler()

        # Hyperparameters: User chosen parameters
        self._epochs = epochs
        self._batch_size = batch_size
        self._timestep = timestep

    # Properties
    @property
    def epochs(self): return self._epochs

    @epochs.setter
    def epochs(self, x): self._epochs = x

    @property
    def batch_size(self): return self._batch_size

    @batch_size.setter
    def batch_size(self, x): self._batch_size = x

    @property
    def timestep(self): return self._timestep

    @timestep.setter
    def timestep(self, x): self._timestep = x

    def create_network(self, x_train, y_train, stock):
        """Creates the network, adds layers then trains the network."""
        model = Sequential()
        model.add(cLSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(cLSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(cLSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(cLSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size)  # Trains the model
        self.myModelHandler.save_model_as_json(model, stock)  # Saves the model as .json and weights as .h5

    def prepare_dataset(self, stock):
        """Reads the stock CSV then creates the training datasets."""
        dataset = pd.DataFrame(yf.download(stock))
        dataset_train = dataset[0:len(dataset)-30]  # Training dataset will be the whole dataset minus the test dataset
        training_set = dataset_train.iloc[:, 1:2].values  # Use the open values only

        # Feature Scaling
        sc = MinMaxScaler(feature_range=(0, 1))  # Scales all values so that they are in the range (0, 1)
        training_set_scaled = sc.fit_transform(training_set)

        # Create an input data structure with a timestep
        x_train = []
        y_train = []
        for i in range(self.timestep, len(dataset)-30):
            x_train.append(training_set_scaled[i - self.timestep:i, 0])
            y_train.append(training_set_scaled[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)

        # Reshaping
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        return x_train, y_train

    def testprepare_dataset(self, stock):
        """Reads the stock CSV then creates the training datasets."""
        dataset = pd.DataFrame(yf.download(stock))
        dataset['Difference'] = dataset['Close'].diff()  # Finds the difference in price each day
        dataset = dataset[1:]  # Drops the first null value
        dataset_train = dataset[0:len(dataset)-30]  # Training dataset will be the whole dataset minus the test dataset
        training_set = dataset_train.iloc[:, 6:7].values  # Use the open values only

        # Feature Scaling
        sc = MinMaxScaler(feature_range=(0, 1))  # Scales all values so that they are in the range (0, 1)
        training_set_scaled = sc.fit_transform(training_set)

        # Create an input data structure with a timestep
        x_train = []
        y_train = []
        for i in range(self.timestep, len(dataset)-30):
            x_train.append(training_set_scaled[i - self.timestep:i, 0])
            y_train.append(training_set_scaled[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)

        # Reshaping
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        return x_train, y_train

    def forecast(self, stock):
        """Loads the trained model, forecasts the values then displays a graph of the values"""
        # Load the trained model
        model_handler = ModelHandler()
        model = model_handler.load_json_model(stock)

        # Importing the training set
        dataset = pd.read_csv(stock.csv_name)
        dates = dataset.iloc[len(dataset)-31:len(dataset)-1, 0].values
        dates = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in dates]

        # Create the test dataset
        dataset_test = dataset[len(dataset) - 30:]
        real_stock_price = dataset_test.iloc[:, 1:2].values
        dataset = dataset['Open']
        inputs = dataset[len(dataset) - len(dataset_test) - 60:].values
        inputs = inputs.reshape(-1, 1)

        # Feature Scaling
        sc = MinMaxScaler(feature_range=(0, 1))
        inputs = sc.fit_transform(inputs)

        x_test = []
        x_test.append(inputs[0:60, 0])
        predicted_values = []
        for i in range(1, 31):
            x_test_np = np.array(x_test)
            x_test_np = np.reshape(x_test_np, (x_test_np.shape[0], x_test_np.shape[1], 1))
            new_data = model.predict(x_test_np)
            predicted_values.append(new_data[0])
            x_test[0] = np.delete(x_test[0], 0)
            x_test[0] = np.concatenate([x_test[0], new_data[0]])

        predicted_values = sc.inverse_transform(predicted_values)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.plot(dates, real_stock_price, color='red', label=f'Actual {stock.ticker} Stock Price')
        plt.plot(dates, predicted_values, color='blue', label=f'Predicted {stock.ticker} Stock Price')
        plt.gcf().autofmt_xdate()
        plt.title(f'{stock.ticker} Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel(f'{stock.ticker} Stock Price')
        plt.legend()
        plt.show()

    def predict(self, stock):
        from model_handler import ModelHandler
        import matplotlib.dates as mdates
        import datetime as dt
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd

        myModelHandler = ModelHandler()
        regressor = myModelHandler.load_json_model(stock)

        # Importing the training set
        dataset = yf.download(stock)

        # Feature Scaling
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler(feature_range=(0, 1))

        dataset_test = dataset[len(dataset)-30:]
        real_stock_price = dataset_test.iloc[:, 1:2].values

        # Getting the predicted stock price
        dataset = dataset['Open']
        inputs = dataset[len(dataset) - len(dataset_test) - 60:].values
        inputs = inputs.reshape(-1, 1)
        inputs = sc.fit_transform(inputs)

        X_test = []
        for i in range(60, 90):
            X_test.append(inputs[i - 30:i, 0])

        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        predicted_stock_price = regressor.predict(X_test)
        print(predicted_stock_price)

        predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        # plt.plot(real_stock_price, color='red', label=f'Actual {stock} Stock Price')
        plt.plot(predicted_stock_price, color='blue', label=f'Predicted {stock} Stock Price')
        plt.title(f'{stock} Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel(f'{stock} Stock Price')
        plt.legend()
        plt.show()

    def testpredict(self, stock):
        from model_handler import ModelHandler
        import numpy as np
        import matplotlib.pyplot as plt

        myModelHandler = ModelHandler()
        regressor = myModelHandler.load_json_model(stock)

        # Importing the training set
        dataset = yf.download(stock)

        # Feature Scaling
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler(feature_range=(0, 1))

        dataset_test = dataset[len(dataset)-30:]
        dataset_test['Difference'] = dataset_test['Open'].diff()
        real_stock_price = dataset_test.iloc[:, 6:7].values

        # Getting the predicted stock price
        dataset['Difference'] = dataset['Close'].diff()
        dataset = dataset['Difference']
        inputs = dataset[len(dataset) - len(dataset_test) - 60:].values
        inputs = inputs.reshape(-1, 1)
        inputs = sc.fit_transform(inputs)

        X_test = []
        for i in range(60, 90):
            X_test.append(inputs[i - 30:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_stock_price = regressor.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        plt.plot(real_stock_price, color='red', label=f'Actual {stock} Stock Price')
        plt.plot(predicted_stock_price, color='blue', label=f'Predicted {stock} Stock Price')
        plt.title(f'{stock} Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel(f'{stock} Stock Price')
        plt.legend()
        plt.show()
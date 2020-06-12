import yfinance as yf
from sklearn import preprocessing
from collections import deque
import numpy as np
import random
# from imblearn.over_sampling import SMOTE
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, CuDNNLSTM, BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
import keras
from model_handler import ModelHandler
from keras.models import load_model


class LstmNetwork:
    """An LSTM network to predict future stock values."""
    def __init__(self, stock, timestep=60, batch_size=64, period=3, epochs=15, split_threshold=0.05):
        self.stock = stock  # Stock ticker as string.
        self.timestep = timestep  # How much data to use to predict values.
        self.batch_size = batch_size  # How much data to pass through the network at once.
        self.period = period  # How many days in the future to predict.
        self.epochs = epochs  # How many times the model will be optimised.
        self.split_threshold = split_threshold  # How much data will be used of validation.
        self.name = f"{self.stock}-{self.period}"  # STOCK-PERIOD

    def classify(self, current, future):
        """If the predicted price is larger than the current price, return 1. Else, return 0."""
        if float(future) > float(current):
            return 1  # 1 denotes that the price is predicted to increase.
        else:
            return 0  # 0 denotes that the price is predicted to decrease.

    def create_datasets(self):
        """Returns the training and validation dataset."""
        # Creating the dataset.
        df = yf.download(self.stock)  # Download the stock dataset.
        #df = df.loc[:, ['Close', 'Volume']]  # Use only Close and Volume data.
        df['Future'] = df['Close'].shift(-self.period)  # Future is the Close price in FUTURE_PERIOD_PREDICT days.
        df['Target'] = list(map(self.classify, df['Close'], df['Future']))

        # Splitting the dataset into training and validation datasets.
        times = sorted(df.index.values)  # Numpy array of the sorted date indices in the dataset.
        split_threshold = times[-int(0.05*len(times))]  # Gets the last 5% threshold index.
        validation_df = df[(df.index >= split_threshold)]  # Creates validation dataset using threshold.
        training_df = df[(df.index < split_threshold)]  # Creates training dataset using threshold.

        return training_df, validation_df

    def create_input_data(self, dataset, balance=False):
        """Returns data that can be inputted into the model"""
        dataset = dataset.drop('Future', 1)  # Removes the Future column from the dataset.

        for column in dataset.columns:  # Loops through each dataset column.
            if column != 'Target':  # We don't want to change the Target column.
                dataset.dropna(inplace=True)  # Removes any NaN values.
                # NOTE: MAY CAUSE BIAS.
                dataset[column] = preprocessing.scale(dataset[column].values)  # Scales the values in range 0, 1.

        dataset.dropna(inplace=True)  # Removes any NaN values created from scaling values.
        sequential_data = []
        prev_days = deque(maxlen=self.timestep)  # Creates a queue with max length TIMESTEP.

        for row in dataset.values:  # row will be a list containing data from a single row.
            prev_days.append([x for x in row[:-1]])  # Append a list of data to prev_days excluding Target value.
            if len(prev_days) == self.timestep:  # If the prev_days queue is full.
                sequential_data.append([np.array(prev_days), row[-1]])  # Appends list of 60 prev_days and Target.

        # NOTE: SHUFFLING DATA MAY DECREASE ACCURACY.
        # random.shuffle(sequential_data)  # Shuffles data.

        # Balancing data.
        buys = []  # Appends sequence and target if the price is predicted to increase.
        sells = []  # Appends sequence and target if the price is predicted to decrease.

        for sequence, target in sequential_data:
            if target == 0:
                sells.append([sequence, target])
            elif target == 1:
                buys.append([sequence, target])

        lower = min(len(buys), len(sells))  # Identifies the minority class.

        # NOTE: THIS METHOD MAY CAUSE UNDERSAMPLING.
        if balance:
            buys = buys[:lower]
            sells = sells[:lower]
            sequential_data = buys+sells
            random.shuffle(sequential_data)

        x = []  # Creates input dataset
        y = []  # Creates output dataset

        for sequence, target_value in sequential_data:
            x.append(sequence)
            y.append(target_value)

        return np.array(x), y

    def train_model(self):
        """Creates and trains the LSTM model."""
        # Creating the input data.
        training_df, validation_df = self.create_datasets()  # Create datasets.
        train_x, train_y = self.create_input_data(training_df)  # Create input values.
        validation_x, validation_y = self.create_input_data(validation_df)

        # Creating the model.
        model = Sequential()
        model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(CuDNNLSTM(128, return_sequences=True))
        model.add(Dropout(0.1))
        model.add(BatchNormalization())
        model.add(CuDNNLSTM(128))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        # Define optimiser.
        opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)

        # Compile model.
        model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        # Train model.
        model.fit(
            train_x, train_y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(validation_x, validation_y)) 

        # Save model as "STOCK-PERIOD.h5"
        model.save(f"models/{self.stock}-{str(self.period)}.h5")
    
    def create_predict_datasets(self):
        df = yf.download(self.stock)  # Download the stock dataset.
        df = df.iloc[-60:, :]  # Selects last 63 days (3 values will be NaN so there will be 60 days worth of data.)
        df = df.loc[:, ['Close', 'Volume']]  # Use only Close and Volume data.
        df['Future'] = df['Close'].shift(-self.period)  # Future is the Close price in FUTURE_PERIOD_PREDICT days.
        df['Target'] = list(map(self.classify, df['Close'], df['Future']))

        return df
    
    def create_predict_input_data(self):
        df = yf.download(self.stock)  # Download the stock dataset.
        df = df.iloc[-62:, :]  # Selects last 63 days (3 values will be NaN so there will be 60 days worth of data.)
        df = df.loc[:, ['Close', 'Volume']]  # Use only Close and Volume data.
        for column in df.columns:  # Loops through each dataset column.
            df[column] = df[column].pct_change()  # Converts values to percentage changes.
            df.dropna(inplace=True)  # Removes any NaN values.
            # NOTE: MAY CAUSE BIAS.
            df[column] = preprocessing.scale(df[column].values)  # Scales the values in range 0, 1.
        return np.array([df.values])
        
    def predict(self):
        model = load_model('models/TSLA-3.h5')  # Loads the model.
        predict_x = self.create_predict_input_data()
        prediction = model.predict(predict_x)
        print(prediction)

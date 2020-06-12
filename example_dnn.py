import yfinance as yf
from sklearn import preprocessing
from collections import deque
import numpy as np
import random
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, CuDNNLSTM, BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint
import keras

# Constants.
TIMESTEP = 60  # How much data to use when predicting.
FUTURE_PERIOD_PREDICT = 3  # How many days to predict.
STOCK_TO_PREDICT = 'MNG'  # The stock ticker that references the stock to predict.
EPOCHS = 20
BATCH_SIZE = 64
NAME = f"Stock-{STOCK_TO_PREDICT}-Timestep-{TIMESTEP}-Period-{FUTURE_PERIOD_PREDICT}-Epochs-{EPOCHS}-" \
       f"Batch-{BATCH_SIZE}-Time-{int(time.time())}"


# Functions.
def classify(current, future):
    """If the predicted price is larger than the current price, return 1. Else, return 0."""
    if float(future) > float(current):
        return 1  # 1 denotes that the price is predicted to increase.
    else:
        return 0  # 0 denotes that the price is predicted to decrease.


def preprocess_df(dataset, balance=False):
    dataset = dataset.drop('Future', 1)  # Removes the Future column from the dataset.

    for column in dataset.columns:  # Loops through each dataset column.
        if column != 'Target':  # We don't want to change the Target column.
            #dataset[column] = dataset[column].pct_change()  # Converts values to percentage changes.
            dataset.dropna(inplace=True)  # Removes any NaN values.
            # NOTE: MAY CAUSE BIAS.
            dataset[column] = preprocessing.scale(dataset[column].values)  # Scales the values in range 0, 1.

    dataset.dropna(inplace=True)  # Removes any NaN values created from scaling values.
    sequential_data = []
    prev_days = deque(maxlen=TIMESTEP)  # Creates a queue with max length TIMESTEP.

    for row in dataset.values:  # row will be a list containing data from a single row.
        prev_days.append([x for x in row[:-1]])  # Append a list of data to prev_days excluding Target value.
        if len(prev_days) == TIMESTEP:  # If the prev_days queue is full.
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


# Creating the dataset.
df = yf.download(STOCK_TO_PREDICT)  # Download the stock dataset.
#df = df.loc[:, ['Close', 'Volume']]  # Use only Close and Volume data.
df['Future'] = df['Close'].shift(-FUTURE_PERIOD_PREDICT)  # Future is the Close price in FUTURE_PERIOD_PREDICT days.
df['Target'] = list(map(classify, df['Close'], df['Future']))  # Target will be the result of classify function.

# Splitting the dataset into training and validation datasets.
times = sorted(df.index.values)  # Numpy array of the sorted date indices in the dataset.
split_threshold = times[-int(0.05*len(times))]  # Gets the last 5% threshold index.
validation_df = df[(df.index >= split_threshold)]  # Creates validation dataset using threshold.
training_df = df[(df.index < split_threshold)]  # Creates training dataset using threshold.
train_x, train_y = preprocess_df(training_df)
validation_x, validation_y = preprocess_df(validation_df)

print(train_x)
# Creating the model.
model = Sequential()
model.add(CuDNNLSTM(50, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(CuDNNLSTM(50, return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(CuDNNLSTM(50))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                                      mode='max'))

history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y))

predictions = model.predict(validation_x)
correct = 0
for pred, y in zip(predictions, validation_y):
    if pred > 0.5:
        pred = 1.0
    else:
        pred = 0.0
    if pred == y:
        correct += 1
    print(f'Prediction: {pred}  -  Actual: {y}')

print(f'Accuracy: {correct/len(predictions)*100}')

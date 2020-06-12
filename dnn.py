import numpy as np
import yfinance as yf
import random


class Dnn:
    """
    A neural network with a variable amount of layers using a single activation function. Minimises the cost function
    using stochastic gradient descent.
    """
    def __init__(self, stock=None, learning_rate=0.05, decay=0.01, epochs=1, layers=[5, 1], batch_size=32, timestep=60, period=1, print_var=True):
        # Define stock to predict.
        self.stock = stock.upper()  # The stock ticker to predict as a string.

        # Define hyper-parameters.
        self.learning_rate = learning_rate  # Defines how much we change the parameters after each propagation.
        self.decay = decay  # How quickly the learning rate gets smaller.
        self.epochs = epochs  # Defines how many times we will pass the data through the network.
        self.batch_size = batch_size  # How many examples to use in order to update the parameters.
        self.timestep = timestep  # How many days of data to use in order to make a prediction.
        self.period = period  # How many days in the future to forecast.

        # Define structure.
        self.layers = [self.timestep] + layers  # A list containing the number of neurons in each layer.

        # Define weights:
        # Weights are of size - (next_layer, previous_layer)
        self.weights = []
        for x in range(len(self.layers)-1):
            self.weights.append(np.random.random((self.layers[x+1], self.layers[x])))
        
        # Define biases.
        # Biases are of size - (layer, 1)
        self.biases = []
        for x in self.layers[1:]:
            self.biases.append(np.random.random((x, 1)))
        
        # Cache stores past data as dict.
        self.cache = {}
        self.cache['cost'] = 100
        
        self.print_var = print_var

    def predict(self):
        df = yf.download(self.stock)
        df = df['Close']
        df = df.values
        df = np.array(df[-self.timestep:]).reshape(self.timestep, 1)
        accuracy = self.train()
        activations = self.predict_feed_forward(df)
        
        prediction = activations[-1]  # Gets the output of the network
        
        if prediction >= 0.5:
            prediction = 1.0
        else:
            prediction = 0.0

        return [prediction, accuracy]

    def prepare_dataset(self):
        """Creates the inputs and desired outputs for the neural network."""
        df = yf.download(self.stock)  # Downloads the stock data from yahoo finance.
        df = df['Close']  # Takes only the close price.
        df = df.values
        future_values = []
        x = []
        y = []
        
        for i in range(self.timestep, len(df) - self.period + 1):
            x.append(np.array(df[i - self.timestep : i]).reshape(self.timestep, 1))
            future_values.append(df[i + self.period - 1])
        
        for i in range(len(x)):
            if future_values[i] > x[i][-1]:
                y.append(1.0)  # If the price goes up, append 1.0.
            else:
                y.append(0.0)
        
        self.num_examples = len(x)
        
        return x, y
    
    def create_batches(self, x, y):
        x_batches = [x[i * self.batch_size : (i + 1) * self.batch_size] 
                     for i in range((len(x) + self.batch_size - 1) // self.batch_size)]
        y_batches = [y[i * self.batch_size : (i + 1) * self.batch_size] 
                     for i in range((len(y) + self.batch_size - 1) // self.batch_size)]
        
        return x_batches, y_batches

    def train(self):
        learning_rate_0 = self.learning_rate  # Initial learning rate.
        x, y = self.prepare_dataset()
        x_batches, y_batches = self.create_batches(x, y)
        for epoch in range(self.epochs):
            cost = self.one_epoch(x_batches, y_batches)
            self.learning_rate = learning_rate_0 * (1 / (1 + self.decay * epoch))  # Makes the learning rate smaller.
            if self.print_var:
                print(f'EPOCH {epoch+1}:     COST = {cost[0][0]}')
            
            # Checks if the cost has increased.
            if self.cache['cost'] < cost:
                break

            self.cache['cost'] = cost

        accuracy = 0
        for xs, ys in zip(x, y):
            cost, activations, net_input = self.feed_forward(xs, ys)
            activation = activations[-1][0][0]
            if round(activation) == ys:
                accuracy += 1

        return round(accuracy / self.num_examples * 100)

    def one_epoch(self, x_batches, y_batches):
        """Trains the neural network for one epoch."""
        cost = 0
        for x_batch, y_batch in zip(x_batches, y_batches):
            cost += self.update_parameters(x_batch, y_batch)  # Updates the weights and biases

        return cost / self.num_examples  # Returns the average cost for the epoch.
        
    def update_parameters(self, x_batch, y_batch):
        """
        Updates the weights and biases after passing a batch through the network:
        The sum of the weight and bias gradients are taken for each example and are then averaged
        to get an average weight and bias update.
        The weight and bias gradients are subtracted from the weights and biases to update the parameters.
        The learning rate changes how much the gradients affect the parameters.
        """
        weight_gradients = [np.zeros(weight.shape) for weight in self.weights]
        bias_gradients = [np.zeros(bias.shape) for bias in self.biases]
        cost = 0
        
        for x, y in zip(x_batch, y_batch):
            new_cost, activations, net_inputs = self.feed_forward(x, y)
            cost += new_cost
            new_weight_gradients, new_bias_gradients = self.backpropagation(activations, net_inputs, y)
            weight_gradients = [a + b for a, b in zip(weight_gradients, new_weight_gradients)]
            bias_gradients = [a + b for a, b in zip(bias_gradients, new_bias_gradients)]

        # Calculate average weight and bias changes.
        weight_gradients = [w * self.learning_rate / len(x_batch) for w in weight_gradients]
        bias_gradients = [b * self.learning_rate / len(x_batch) for b in bias_gradients]

        # Update parameters using: w_new = w - gradient * learning rate
        self.weights = [w - g for w, g in zip(self.weights, weight_gradients)]
        self.biases = [b - g for b, g in zip(self.biases, bias_gradients)]

        return cost

    def check_parameters(self, x_batch, y_batch):
        cost = 0

        for x, y in zip(x_batch, y_batch):
            new_cost, activations, net_inputs = self.feed_forward(x, y)
            cost += new_cost
        
        return cost
        
    def neuron(self, w, b, x):
        """Applies weights and biases to input and passes through sigmoid function. Returns activation."""
        a = np.matmul(w, x)  # Matrix multiplies input and weights.
        net_input = a + b  # Adds biases.
        a = self.sigmoid(net_input)  # Passes through sigmoid activation.

        return a, net_input

    def feed_forward(self, x, y):
        """
        Performs one forward pass in the neural network i.e. passes input data through each layer.
        """
        activations = [x]  # To be used during backpropagation.
        net_inputs = []  # To be used during backpropagation.
        a = x  # The first activation will be the x values.
        for weight, bias in zip(self.weights, self.biases):
            a, net_input = self.neuron(weight, bias, a)  # Passes through sigmoid neuron.
            activations.append(a)
            net_inputs.append(net_input)
        cost = self.mean_squared_error(y, activations[-1])  # Calculates the cost

        return cost, activations, net_inputs

    def predict_feed_forward(self, x):
        activations = []
        a = x

        for weight, bias in zip(self.weights, self.biases):
            a, net_input = self.neuron(weight, bias, a)  # Passes through sigmoid neuron.
            activations.append(a)

        return activations

    def backpropagation(self, activations, net_inputs, y):
        # Initialize weight and bias gradients with the same size as their respective parameters.
        weight_gradients = [np.zeros_like(weight) for weight in self.weights]
        bias_gradients = [np.zeros_like(bias) for bias in self.biases]

        # Calculates the delta of the last layer.
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(net_inputs[-1])
        bias_gradients[-1] = delta
        weight_gradients[-1] = np.matmul(delta, activations[-2].transpose())

        # Finds the gradient of the function with respect to every weight and bias.
        for l in range(2, len(self.layers)):
            net = net_inputs[-l]
            sp = self.sigmoid_prime(net)
            delta = np.matmul(self.weights[-l+1].transpose(), delta) * sp
            bias_gradients[-l] = delta
            weight_gradients[-l] = np.matmul(delta, activations[-l-1].transpose())
        
        return weight_gradients, bias_gradients
    
    def mean_squared_error(self, y, o):
        """
        Calculates the mean squared error:
        Since the neural network only has one output, y (desired output) and o (network output)
        will be a single value.
        This means that the total cost will also be one value.
        """
        cost = 1/2*(y - o)**2

        return cost

    def sigmoid(self, x):
        """
        The activation function that is used in this neural network.
        """
        return 1 / (1 + np.exp(-x))
    
    def cost_derivative(self, y, o):
        """
        Used during back-propagation:
        We need to work out dC / dO (change in cost with respect to output).
        Since C = 1/2 * (y - o)**2
        Using differentiation:
        dC = y - o
        """

        dC = y - o

        return dC
        
    def sigmoid_prime(self, o):
        """
        Used during back-propagation:
        We need to work out dO / dnet (change in output with respect to net input).
        Since O = sigmoid(net)
        Using differentiation:
        dO = sigmoid_prime(O)
        """

        dO = self.sigmoid(o) * (1 - self.sigmoid(o))

        return dO

import random
import numpy as np
import matplotlib.pyplot as plt


class Layer:
    def __init__(self, input_num: int, output_num: int):
        # Input node number and output node number of the layer
        self.input_num, self.output_num = input_num, output_num
        self.input_vector, self.output_vector = None, None
        # Delta weight and delta bias
        self.d_weight, self.d_bias = None, None
        # Generate weight matrix and bias for normal distribution
        self.weight = np.random.normal(loc=0, scale=0.2, size=(input_num, output_num))
        # self.weight /= np.sqrt(input_num + output_num)
        self.bias = np.random.normal(loc=0, scale=0.2, size=(1, output_num))

    # Compute process in every layer
    def forward(self, vector: np.ndarray) -> np.ndarray:
        self.input_vector = vector
        temp = self.bias + np.matmul(self.input_vector, self.weight)
        self.output_vector = 1.0 / (1.0 + np.exp(-temp))
        return self.output_vector

    # Get delta weight and delta bias, then to the next layer
    def backward(self, grad: np.ndarray) -> np.ndarray:
        grad = grad * self.output_vector * (1 - self.output_vector)
        self.d_weight = np.matmul(self.input_vector.T, grad)
        self.d_bias = grad
        return np.matmul(grad, self.weight.T)

    # Modify weight and bias
    def learn(self, learn_rate: float):
        self.weight -= learn_rate * self.d_weight
        self.bias = self.bias - learn_rate * self.d_bias


# The last layer which does not need to apply sigmoid function
class OutputLayer(Layer):
    # Override forward function
    def forward(self, vector: np.ndarray) -> np.ndarray:
        self.input_vector = vector
        self.output_vector = self.bias + np.matmul(self.input_vector, self.weight)
        return self.output_vector

    # Override backward function
    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.d_weight = np.matmul(self.input_vector.T, grad)
        self.d_bias = grad
        return np.matmul(grad, self.weight.T)


# Loss function of prediction
def mean_squared_error(output: np.ndarray, target: np.ndarray) -> (float, np.ndarray):
    # Compute gradient
    grad = output - target
    # Compute loss
    loss = 0.5 * ((output - target) ** 2)
    return loss, grad


class Network:
    # Generate network by input
    def __init__(self, network: np.ndarray, learn_rate: float):
        self.layers = []
        self.learn_rate = learn_rate
        self.kinds = 12
        i = 0
        while i < len(network) - 2:
            self.layers.append(Layer(network[i], network[i + 1]))
            i += 1
        self.layers.append(OutputLayer(network[-2], 1))

    # Forward propagation
    def forward(self, vector: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            vector = layer.forward(vector)
        return vector

    # backward propagation
    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    # Adjust weights and bias
    def learn(self):
        for layer in self.layers:
            layer.learn(self.learn_rate)

    # Train the network
    def predict(self) -> np.ndarray:
        random.seed(0)
        x = []
        for i in range(300):
            x.append(2 * np.pi * random.random() - np.pi)
        x = np.sort(x)
        reshape_x = np.reshape(x, (-1, 1))
        target = np.sin(reshape_x)
        loss, epoch = 0, 0
        for epoch in range(5000):
            output = self.forward(reshape_x)
            temp_loss, grad = mean_squared_error(output, target)
            loss += temp_loss
            self.backward(grad)
            self.learn()
            if epoch % 500 == 0:
                print(f"epoch {epoch}: loss {np.mean(loss)}")
                loss = 0
        return x

    # Draw function graph
    def draw(self, x: np.ndarray):
        target = np.sin(x)
        predict = self.forward(np.reshape(x, (-1, 1)))
        plt.plot(x, target, label="sin(x)")
        plt.plot(x, predict, ':', label="predict")
        plt.legend()
        plt.show()

    # Function for debugging
    def struct(self):
        print("Network had been generated as:")
        for layer in self.layers:
            print(f"{layer.__class__}: input {layer.input_num}; output {layer.output_num}")
        print(f"And learn rate had been set as {self.learn_rate}")


# Generate the network from input
def generate() -> (np.ndarray, float):
    get_input, get_rate = None, None
    get_input = input("Please generate your network\n"
                      "Every number represent node number of a layer, like 1 32 32 1\n"
                      "And the first and the last layer must be 1 node:\n")
    get_rate = float(input("Please input learning rate:\n"))
    return np.asarray([int(n) for n in get_input.split()]), get_rate


if __name__ == '__main__':
    net, learn_rate = generate()
    network = Network(net, learn_rate)
    network.struct()
    data = network.predict()
    network.draw(data)

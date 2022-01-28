import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import torch
import os


class Layer:
    def __init__(self, input_num: int, output_num: int):
        # Input node number and output node number of the layer
        self.input_num, self.output_num = input_num, output_num
        self.input_vector, self.output_vector = None, None
        # Delta weight and delta bias
        self.d_weight, self.d_bias = None, None
        # Generate weight matrix and bias for normal distribution
        self.weight = np.random.normal(loc=0, scale=1, size=(input_num, output_num))
        self.bias = np.random.normal(loc=0, scale=1, size=(1, output_num))

    # Compute process in every layer
    def forward(self, vector: np.ndarray) -> np.ndarray:
        self.input_vector = vector
        self.output_vector = self.bias + np.dot(self.input_vector, self.weight)
        return self.output_vector

    # Get delta weight and delta bias, then to the next layer
    def backward(self, grad: np.ndarray) -> np.ndarray:
        # Gradient of Sigmoid function
        self.d_weight = np.matmul(self.input_vector.T, grad)
        self.d_bias = np.average(grad, 0)
        return np.matmul(grad, self.weight.T)

    # Modify weight and bias
    def learn(self, learn_rate: float):
        self.weight -= learn_rate * self.d_weight
        self.bias = self.bias - learn_rate * self.d_bias


class Sigmoid:
    def __init__(self, input_num: int, output_num: int):
        self.input_num = input_num
        self.output_num = output_num
        self.output_vector = None
        pass

    def forward(self, vector: int) -> np.ndarray:
        self.output_vector = 1.0 / (1.0 + np.exp(-vector))
        return self.output_vector

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return self.output_vector * (1 - self.output_vector) * grad

    def learn(self, learn_rate: float):
        pass


# Softmax Layer
class Softmax:
    def __init__(self, kinds: int):
        self.input_num = self.output_num = kinds
        self.output = []

    def forward(self, vector: np.ndarray) -> np.ndarray:
        self.output = []
        exp_vector = np.exp(vector)
        i = 0
        while i < (exp_vector.size / 12):
            self.output.append(exp_vector[i] / exp_vector[i].sum())
            i += 1
        return np.asarray(self.output)

    # This function do nothing, as the gradient of cross entropy and
    # softmax had been computed in cross_entropy function
    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad

    # Nothing to do
    def learn(self, learn_rate: int):
        pass


# Loss function of prediction
def mean_squared_error(output: np.ndarray, target: np.ndarray) -> (float, np.ndarray):
    # Compute gradient
    grad = output - target
    # Compute loss
    loss = 0.5 * ((output - target) ** 2)
    return loss, grad


# Loss function of classification
def cross_entropy(output: np.ndarray, target: np.ndarray) -> (float, np.ndarray):
    loss = -1 * np.mean(np.sum(target * np.log(output), axis=1)) / 12
    # Gradient of cross entropy and softmax
    grad = output - target
    return loss, grad


# Transfer image to vector
def image_to_matrix(tag: int) -> (np.ndarray, np.ndarray):
    data, input_vector, label_vector = [], [], []
    label = 1
    while label <= 12:
        # Tag == 0: train
        if tag == 0:
            img = 1
            max_num = 620
        # Tag == 1: test
        else:
            img = 601
            max_num = 620
        while img <= max_num:
            # "train/5/604.bmp" had been deleted
            if label == 5 and img == 604:
                img += 1
                continue
            path = "train/" + str(label) + "/" + str(img) + ".bmp"
            image = Image.open(path)
            temp = np.array(image).astype(int).flatten()
            data.append([temp, label])
            img += 1
        label += 1
    random.shuffle(data)
    i = 0
    while i < len(data):
        input_vector.append(data[i][0])
        label_vector.append(data[i][1])
        i += 1
    return np.asarray(input_vector), np.asarray(label_vector)


class Network:
    # Generate network by input
    def __init__(self, network: np.ndarray, learn_rate: float, mission: str):
        self.layers = []
        self.learn_rate = learn_rate
        self.kinds = 12
        i = 0
        while i < len(network) - 2:
            self.layers.append(Layer(network[i], network[i + 1]))
            self.layers.append(Sigmoid(network[i + 1], network[i + 1]))
            i += 1

        if mission == "P":
            self.layers.append(Layer(network[-2], network[-1]))
        else:
            self.layers.append(Layer(network[-2], self.kinds))
            self.layers.append(Sigmoid(network[i + 1], network[i + 1]))
            self.layers.append(Softmax(self.kinds))

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
        for epoch in range(50000):
            output = self.forward(reshape_x)
            temp_loss, grad = mean_squared_error(output, target)
            loss += temp_loss
            self.backward(grad)
            self.learn()
            if epoch % 5000 == 0:
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

    # Classification function
    def classify(self, path: str):
        input_vector, label_vector = image_to_matrix(0)
        flag = True
        epoch, history_accuracy = 0, 0.0
        while flag:
            cnt, loss = 0, 0
            batch_num = 30
            for i in range(batch_num):
                batch_size = 6840 / batch_num
                input_batch = input_vector[int(i * batch_size): int((i + 1) * batch_size)]
                output = self.forward(input_batch)

                target = []
                for label in label_vector[int(i * batch_size): int((i + 1) * batch_size)]:
                    temp = np.zeros(self.kinds, dtype=int)
                    temp[label - 1] = 1
                    target.append(temp)
                target = np.asarray(target)

                temp_loss, grad = cross_entropy(output, target)
                loss += temp_loss
                i = 0
                while i < batch_size:
                    result = np.argmax(output[i])
                    if np.argmax(target[i]) == result:
                        cnt += 1
                    i += 1
                self.backward(grad)
                self.learn()
            accuracy = cnt / 68.4
            if epoch == 0:
                history_accuracy = accuracy
            if epoch % 200 == 0 and epoch > 0:
                if accuracy == history_accuracy:
                    flag = False
                    print("Accuracy does not rise")
                else:
                    history_accuracy = accuracy
            print("epoch %i: accuracy %.5f%%; loss %.8f" % (epoch, accuracy, loss))
            epoch += 1

        save_path = path
        torch.save(self, save_path)
        print("successfully saved")

    # Test function
    def test(self):
        accuracy_cnt = 0
        input_vector, label_vector = image_to_matrix(1)
        output = self.forward(input_vector)
        i = 0
        while i < 239:
            result = np.argmax(output[i]) + 1
            if result == label_vector[i]:
                accuracy_cnt += 1
            i += 1
        # 600 images had been tested totally
        accuracy = accuracy_cnt / 2.39
        print(f"test accuracy: {accuracy}%")

    # Struct of network, used in debugging
    def struct(self):
        for layer in self.layers:
            print(f"{layer.__class__}: input {layer.input_num}; output {layer.output_num}")


# Generate the network from input
def generate() -> (np.ndarray, float, str):
    get_net, get_rate, get_mission = None, None, None
    get_mission = input("Predict or Classify? [P/C]\n")
    if get_mission == "P":
        get_net = input("Please generate your network\n"
                      "Every number represent node number of a layer, like 1 32 32 1\n"
                      "And the first and the last layer must be 1 node:\n")
        get_rate = float(input("Please input learning rate:\n"))
    else:
        get_net = input("Please input the model path:\n")
    return get_net, get_rate, get_mission


if __name__ == "__main__":
    # network, learn_rate, mission = generate()
    # if mission == "P":
    #     network = np.asarray([int(n) for n in network.split()])
    #     net = Network(network, learn_rate, mission)
    #     data = net.predict()
    #     net.draw(data)
    # else:
    #     net = torch.load(network)
    #     net.classify(network)
    #      net.test()

    uname = input("Please input USB flash disk name\n")
    path = "/Volumes/" + uname + "/test_data"
    model_num = input("Please input model number(1, 2, 3, 4):\n")
    net = torch.load("model" + model_num + ".pth")
    total, cnt, accuracy = 0, 0, 0.0
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            dir_path = path + "/" + dir
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    image = Image.open(dir_path + "/" + file)
                    output = net.forward(image)
                    if np.argmax(output) + 1 == int(dir):
                        cnt += 1
                    total += 1
        accuracy = cnt / total
        print("Accuracy is " + str(accuracy))


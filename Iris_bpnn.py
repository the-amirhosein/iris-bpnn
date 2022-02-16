from __future__ import division
import math
import random
import pandas as pd
import matplotlib.pylab as pl

flower_labels = {0: 'Iris-setosa',
                 1: 'Iris-versicolor',
                 2: 'Iris-virginica'}
random.seed(0)


# generate random num in [a, b)
def rand(a, b):
    return (b - a) * random.random() + a


# generate matrix of size a*b (without numpy)
def create_matrix(a, b, fill=0.0):
    m = []
    for i in range(a):
        m.append([fill] * b)
    return m


# activation function (sigmoid)
def sigmoid(x):
    return math.tanh(x)


# derivative function of sigmoid activation function
def dsigmoid(y):
    return 1.0 - y ** 2


class NN:
    """ Three-layer backpropagation neural network """

    def __init__(self, ni, nh, no):
        # Input layer, hidden layer, output layer nodes (number)
        self.ni = ni + 1  # add a bias node
        self.nh = nh
        self.no = no

        # activate all nodes of the neural network
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        # build weights
        self.wi = create_matrix(self.ni, self.nh)
        self.wo = create_matrix(self.nh, self.no)
        # set to random
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # build the momentum factor
        self.ci = create_matrix(self.ni, self.nh)
        self.co = create_matrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni - 1:
            raise ValueError('The number of value is not match to the input layer！')

        # input layer activation
        for i in range(self.ni - 1):
            # self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # hidden layer activation
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output layer activation
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        # print self.ao
        return self.ao[:]

    def back_propagate(self, targets, n, m):

        # output layer error calculation
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # hidden layer error calculation
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output layer weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] = self.wo[j][k] + n * change + m * self.co[j][k]
                self.co[j][k] = change

        # update input layer weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] = self.wi[i][j] + n * change + m * self.ci[i][j]
                self.ci[i][j] = change

        # error calculation
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5 * (targets[k] - self.ao[k]) ** 2
        error += 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def test(self, patterns):
        count = 0
        for p in patterns:
            target = flower_labels[(p[1].index(1))]
            result = self.update(p[0])
            index = result.index(max(result))
            print(p[0], ':', target, '->', flower_labels[index])
            count += (target == flower_labels[index])

        accuracy = float(count / len(patterns))
        print('accuracy: %-.9f' % accuracy)

    def weights(self):
        print('input layer weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('output layer weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=1000, n=0.1, m=0.01):
        # n: learning rate
        # m: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.back_propagate(targets, n, m)
            if i % 100 == 0:
                print(' loss {e}'.format(i=i, e=error))

        self.weights()


def iris():
    data = []
    # read dataset
    raw = pd.read_csv('iris.csv')
    raw_data = raw.values
    raw_feature = raw_data[0:, 0:4]
    for i in range(len(raw_feature)):
        ele = [list(raw_feature[i])]
        if raw_data[i][4] == 'Iris-setosa':
            ele.append([1, 0, 0])
        elif raw_data[i][4] == 'Iris-versicolor':
            ele.append([0, 1, 0])
        else:
            ele.append([0, 0, 1])
        data.append(ele)
    random.shuffle(data)
    # print data
    training = data[0:120]
    test = data[121:]
    nn = NN(4, 7, 3)
    nn.train(training, iterations=10000)

    nn.test(test)
    show_plot(raw_data)


def show_plot(data):
    x = data[:, 0]
    y = data[:, 1]
    color_dict = {'Iris-setosa': 'r',
                  'Iris-versicolor': 'g',
                  'Iris-virginica': 'b'}
    pl.scatter(x, y, color=[color_dict[i] for i in data[:, -1]])
    pl.show()


if __name__ == '__main__':
    iris()

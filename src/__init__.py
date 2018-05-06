from builtins import print

import numpy as np
import time
from math import exp


class NN:
    l = 4
    n = 9
    o = None
    b = None
    w = None
    e = None

    def __init__(self):
        self.o = np.zeros((self.l, self.n))
        self.b = np.random.random((self.l, self.n))
        self.w = np.random.random((self.l, self.n, self.n))
        self.e = np.zeros((self.l, self.n))
        np.vectorize(self.sigmoid)
        np.vectorize(self.sigmoid_deriv)

    def run_through_network(self, i):
        self.o[0] = i
        for lx in range(1, self.l):
            self.o[lx] = self.sigmoid(np.sum(self.o[lx-1] * self.w[lx], axis=1) + self.b[lx])

    def backprop_error(self, o_exp):
        o = self.o[self.l-1]
        o_deriv = self.sigmoid_deriv(o)
        e = (o - o_exp) * o_deriv
        self.e[self.l-1] = e

        for lx in range(self.l-2, -1, -1):
            self.e[lx] = np.sum(self.w[lx+1] * self.e[lx+1]) * self.sigmoid_deriv(self.o[lx])

    def update_biases_and_weights(self):
        d = -0.3 * self.e
        self.b += d
        for lx in range(1, self.l):
            self.w[lx] += d[lx]*self.o[lx-1]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return x * (1 - x)


ts = time.time()

board_states = np.zeros((9, 9))
np.fill_diagonal(board_states, 1)
outputs = np.ones((9, 9))
np.fill_diagonal(outputs, 0)

network = NN()
for i in range(10000):
    rn = np.random.randint(0, 2)
    board_state = board_states[rn]
    expected_output = outputs[rn]

    network.run_through_network(board_state)
    network.backprop_error(expected_output)
    network.update_biases_and_weights()

for i in range(9):
    board_state = board_states[i]
    expected_output = outputs[i]
    network.run_through_network(board_state)
    print(np.round(network.o[network.l-1]))

tf = time.time()
print("Execution time: {0}".format(tf - ts))

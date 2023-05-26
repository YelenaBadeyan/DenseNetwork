import numpy as np

class DenseNetwork:
    def __init__(self, input_dim, num_neurons, output_dim, num_layers):
        self.input_dim = input_dim
        self.num_neurons = num_neurons
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.w = []
        self.b = []
        self.gradw = []
        self.gradb = []
        self.E = []

        self.w.append(np.random.randn(self.input_dim, self.num_neurons))
        for i in range(num_layers - 1):
            self.w.append(np.random.randn(self.num_neurons, self.num_neurons))
        self.w.append(np.random.randn(self.num_neurons, self.output_dim))

        self.b.append(np.zeros((1, self.num_neurons)))
        for i in range(num_layers - 1):
            self.b.append(np.zeros((1, self.num_neurons)))
        self.b.append(np.zeros((1, self.output_dim)))

    def activation(self, x):
        x = [-1 * num for num in x]
        return (1) / (1 + np.exp(x))

    def activation_deriv(self, x):
        return self.activation(x) * (1 - self.activation(x))

    def forward(self, x):
        self.z = []
        self.a = []
        self.z.append(x)
        self.a.append(x)

        for i in range(self.num_layers):
            self.z.append(np.dot(self.a[i], self.w[i]) + self.b[i])
            self.a.append(self.activation(self.z[i + 1]))
        return self.a[-1]

    def mse(self, x, y):
        return np.mean(np.power(self.predict(x) - y, 2))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def backward(self, x, y):
        self.gradw = []
        self.gradb = []
        self.E = []

        self.E.append(self.a[-1] - y)

        self.gradw.append(np.dot(self.a[-2].T, self.E[-1]))
        self.gradb.append(np.sum(self.E[-1], axis=0, keepdims=True))

        for i in range(self.num_layers - 1, 0, -1):
            self.E.append(np.dot(self.E[-1], self.w[i].T) * self.activation_deriv(self.z[i]))
            self.gradw.append(np.dot(self.a[i - 1].T, self.E[-1]))
            self.gradb.append(np.sum(self.E[-1], axis=0, keepdims=True))

        self.gradw.reverse()
        self.gradb.reverse()
        self.E.reverse()

    def update(self, lr):
        for i in range(self.num_layers):
            self.w[i] -= lr * self.gradw[i]
            self.b[i] -= lr * self.gradb[i]

    def train(self, x, y, lr, epochs):
        for i in range(epochs):
            self.forward(x)
            self.backward(x, y)
            self.update(lr)
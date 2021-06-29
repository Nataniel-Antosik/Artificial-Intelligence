from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
import matplotlib.pyplot as plt

def coding(y_iris_coded, y_iris, nSamples):
    for i in range(nSamples):
        if y_iris[i] == "Iris-setosa":
            y_iris_coded[i] = [1, 0, 0]
        elif y_iris[i] == "Iris-versicolor":
            y_iris_coded[i] = [0, 1, 0]
        elif y_iris[i] == "Iris-virginica":
            y_iris_coded[i] = [0, 0, 1]

class MLP(object):
    def __init__(self):
        self.hidden = 10
        self.epochs = 100
        self.eta = 0.1
        self.shuffle = True

    def _sigmoid(self, z):
        return (1 / (1 + np.exp(-z)))

    def _forward(self, X):
        o_hidden = X.dot(self.w_h) + self.b_h
        tmp_o_hidden = self._sigmoid(o_hidden)
        o_out = tmp_o_hidden.dot(self.w_out) + self.b_out
        tmp_o_out = self._sigmoid(o_out)
        return tmp_o_hidden, tmp_o_out

    def _compute_cost(self, y, output):
        cost = 0
        for i in range(self.n_samples):
            for k in range(self.n_classes):
                cost += (y[i][k] * np.log(output[i][k])) + ((1 - y[i][k]) * np.log(1 - output[i][k]))
        cost = -cost
        return cost

    def accuracy(self, y_expected, y_test):
        counter = 0
        for i in range(len(y_test)):
            if (y_expected[i] == y_test[i]).all():
                counter += 1
        return ((counter / len(y_test)) * 100)

    def fit(self, X_train, y_train, plot_name):
        self.n_samples = len(y_train)
        self.n_classes = len(np.unique(y_train,axis = 0))
        self.n_parameters = np.shape(X_train)[1]
        self.w_h = np.random.normal(0, 0.1, (self.n_parameters,self.hidden))
        self.b_h = np.zeros(self.hidden)
        self.w_out = np.random.normal(0, 0.1, (self.hidden, self.n_classes))
        self.b_out = np.zeros(self.n_classes)
        self.all_cost = []
        self.all_accuracy = []

        new_tab = np.hstack((X_train, y_train)) #combining to mix indexes

        for i in range(self.epochs):
            random.shuffle(new_tab)
            new_X_train = new_tab[:, [0, 1, 2, 3]]
            new_y_train = new_tab[:, [4, 5, 6]]
            for j in range(self.n_parameters):
                a_hidden, a_out = self._forward(new_X_train)
                tmp_a_out = (a_out * (1 - a_out))
                delta_out = (a_out - new_y_train) * tmp_a_out
                tmp_a_hidden = (a_hidden * (1 - a_hidden))
                delta_h = delta_out.dot(np.transpose(self.w_out)) * tmp_a_hidden

                gradient_w_hidden = np.dot(np.transpose(new_X_train), delta_h)
                gradient_b_hidden = delta_h
                gradient_w_out = np.dot(np.transpose(a_hidden), delta_out)
                gradient_b_out = delta_out

                self.w_h -= gradient_w_hidden * self.eta
                self.b_h -= np.sum(gradient_b_hidden) * self.eta
                self.w_out -= gradient_w_out * self.eta
                self.b_out -= np.sum(gradient_w_out) * self.eta

            self.all_cost.append(self._compute_cost(new_y_train, a_out))
            self.all_accuracy.append(self.accuracy(new_y_train, self.predict(new_X_train)))

        self.plot(self.epochs, self.all_cost, self.all_accuracy, plot_name)

    def predict(self, X):
        max_index = []
        output = []
        new_hidden, new_out = self._forward(X)
        for i in range(len(new_out)):
            max_index.append(np.argmax(new_out[i]))
            if max_index[i] == 2:
                output.append([0, 0, 1])
            elif max_index[i] == 1:
                output.append([0, 1, 0])
            elif max_index[i] == 0:
                output.append([1, 0, 0])
        return output

    def plot(self, epochs, cost, accuracy, plot_name):
        plt.plot(range(epochs), cost)
        plt.xlabel("epochs")
        plt.ylabel("cost")
        plt.title(plot_name)
        plt.savefig(plot_name + '_cost.png', dpi = 600)
        plt.show()
        plt.figure()
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.plot(range(epochs), accuracy)
        plt.title(plot_name)
        plt.savefig(plot_name + '_accuracy.png', dpi = 600)
        plt.show()

if __name__ == '__main__':
    X_iris, y_iris = fetch_openml(name="iris", version=1, return_X_y=True)

    mlp = MLP()
    mlp2 = MLP()

    nClasses = len(set(y_iris))
    nSamples = len(y_iris)

    print("Number of classes: |", nClasses, "| Number of samples: |", nSamples, "| Classes: |", set(y_iris), "|")

    y_iris_coded = np.zeros([nSamples, nClasses])

    coding(y_iris_coded, y_iris, nSamples)
    X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris_coded, random_state=13)

    print("Training values: classes(", len(np.unique(y_train,axis = 0)), ") samples: (", len(y_train), ") features(", np.shape(X_train)[1], ")")
    mlp.fit(X_train, y_train, 'Training values')
    print("Values for test data: ")
    mlp2.fit(X_test, y_test, 'Test data')

    binary_X = np.mod(np.random.permutation(130*4).reshape(130,4),2)
    binary_y = np.mod(np.random.permutation(130*3).reshape(130,3),2)

    print("Binary Data x: ", binary_X)
    print("Binary Data y: ", binary_y)

    binary_X_train, binary_X_test, binary_y_train, binary_y_test = train_test_split(X_iris, y_iris_coded, random_state=13)

    mlp3 = MLP()

    print("Values for binary data: ")
    mlp3.fit(binary_X_train, binary_y_train, 'Binary data')

    s = MinMaxScaler()

    scalet_data = s.fit_transform(X_iris)
    s_X_train, s_X_test, s_y_train, s_y_test = train_test_split(scalet_data, y_iris_coded, random_state=13)

    mlp4 = MLP()
    print("Values for normalized data: ")
    mlp4.fit(s_X_train, s_y_train, 'Normalized data')

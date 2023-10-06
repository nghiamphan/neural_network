import math
import torch
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


def read_data(file_name: str = "data_banknote_authentication.txt", test_size: int = 0.2) -> tuple[list]:
    """
    Read raw data and divide it into training and testing sets.

    Returns
    -------
        tuple[list]
            (train_inputs, test_inputs, train_targets, test_targets)
    """
    raw_matrix = np.genfromtxt(file_name, delimiter=",")
    inputs = raw_matrix[:, :-1]
    targets = raw_matrix[:, -1]
    return train_test_split(inputs, targets, test_size=test_size)


def sklearn_logistic_reg(X_train: list[list], X_test: list[list], y_train: list[float], y_test: list[float]) -> None:
    """
    Train and test the data sets using sklearn's built-in logistic regression.
    """
    lr = LogisticRegression()

    # train the model
    lr.fit(X_train, y_train)

    # calculate the accuracy and loss function of the model on testing data set
    y_predict = lr.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_predict)
    log_loss = metrics.log_loss(y_test, y_predict)
    print("Sklearn Accuracy: ", accuracy, "     Sklearn Log Loss: ", log_loss)


# Neural Network with 0 hidden layer
class NN:
    def __init__(self) -> None:
        return None

    def import_data_from_file(
        self, file_name: str = "data_banknote_authentication.txt", test_size: int = 0.2, normalization: bool = False
    ) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = read_data(file_name, test_size)
        self.setup_nn(normalization)

    def import_data(
        self,
        X_train: list[list],
        X_test: list[list],
        y_train: list[float],
        y_test: list[float],
        normalization: bool = False,
    ) -> None:
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.setup_nn(normalization)

    def setup_nn(self, normalization: bool = False) -> None:
        """
        Normalize the inputs if necessary. Add the bias node for each input vector. Set up weights.
        """
        if normalization:
            self.X_train = preprocessing.normalize(self.X_train)
            self.X_test = preprocessing.normalize(self.X_test)

        bias = [[1]] * len(self.X_train)
        self.X_train = np.hstack((bias, self.X_train))
        self.y_train = np.array(self.y_train)

        bias = [[1]] * len(self.X_test)
        self.X_test = np.hstack((bias, self.X_test))
        self.y_test = np.array(self.y_test)

        self.setup_weight()

    def setup_weight(self):
        self.w = np.random.rand(len(self.X_train[0]))
        self.original_w = np.array(self.w)

    def reset_weight(self):
        """
        This function is called before each training in cross validation to make comparison between different parameters fairer.
        """
        self.w = np.array(self.original_w)

    def sigmoid(self, X: list[float] | list[list], w: list[float]) -> float | list[float]:
        """
        Calculate sigmoid function.

        If X is an 1-d array (representing a single input vector),  return a single value.
        If X is a  2-d array (representing multiple input vectors), return a list of floats.
        """
        z = X @ w
        if X.ndim == 1:
            return 1 / (1 + math.exp(z))
        return np.array([1 / (1 + math.exp(n)) for n in z])

    def forward_propagation(self, X: list[float] | list[list]) -> float | list[float]:
        return self.sigmoid(X, self.w)

    def backward_propagation(self, X: list[float] | list[list], y: int | list[int], l2: float) -> list[float]:
        """
        Parameters
        ----------
        X: list | list[list]
            A single input vector | Multiple input vectors
        y: int | list
            A single target value | A list of target values
        l2: float
            Lambda factor in L2 regularization

        Return
        ------
        dw: list[float]
            List of the gradients of the weights between input and output layers
        """
        y_predict = self.forward_propagation(X)

        if X.ndim == 1:
            dw = X * (y_predict - y) - l2 * self.w
        else:
            dw = X.T @ (y_predict - y) / len(X) - l2 * self.w

        return dw

    def quick_train(
        self,
        X: list[list],
        y: list[int],
        epochs: int = 1,
        learning_rate: float = 1,
        l2: float = 0,
        print_result: bool = False,
    ) -> float:
        """
        Train the neural network and update weights after the forward pass of each input vector.

        Parameters
        ----------
        X: list[list]
            Training data input
        y: list[int]
            Training data target
        epochs: int
            Number of training iterations
        learning_rate: float
        l2: float
            Lambda factor in L2 regularization
        print_result: bool
            Whether to print some result after each training iteration

        Return
        -------
        test_log_loss: float
            Log loss on test data at the end of the training
        """
        scores = []
        self.reset_weight()
        for epoch in range(1, epochs + 1):
            for i in range(len(X)):
                dw = self.backward_propagation(X[i], y[i], l2)
                self.w += learning_rate * dw

            train_accuracy, train_log_loss = self.evaluate(X, y)
            test_accuracy, test_log_loss = self.evaluate(self.X_test, self.y_test)

            if print_result:
                self.print_result(epoch, train_accuracy, train_log_loss, test_accuracy, test_log_loss, self.w)
                scores.append([train_accuracy, train_log_loss, test_accuracy, test_log_loss])

        if print_result:
            self.draw_result(scores)

        return test_log_loss

    def slow_train(self, X, y, epochs=1, learning_rate=1, l2=0, print_result=False) -> float:
        """
        Train the neural network and update weights after each entire epoch forward pass.

        Parameters & Return
        -------------------
        Same as method quick_train()
        """
        scores = []
        self.reset_weight()
        for epoch in range(1, epochs + 1):
            dw = self.backward_propagation(X, y, l2)
            self.w += learning_rate * dw

            train_accuracy, train_log_loss = self.evaluate(X, y)
            test_accuracy, test_log_loss = self.evaluate(self.X_test, self.y_test)

            if print_result:
                self.print_result(epoch, train_accuracy, train_log_loss, test_accuracy, test_log_loss, self.w)
                scores.append([train_accuracy, train_log_loss, test_accuracy, test_log_loss])

        if print_result:
            self.draw_result(scores)

        return test_log_loss

    def cross_validation(self, k_fold: int, epochs: int, learning_rate: float, l2: float) -> None:
        """
        Train the neural network with cross validation and report the average log loss.

        Parameters
        ----------
        k_fold:
            Number of folds
        epochs: int
            Number of training iterations
        learning_rate: float
        l2: float
            Lambda factor in L2 regularization
        """
        test_log_loss = 0
        for i in range(k_fold):
            n = len(self.X_train) // k_fold

            X = np.vstack((self.X_train[: i * n], self.X_train[(i + 1) * n :]))
            y = np.hstack((self.y_train[: i * n], self.y_train[(i + 1) * n :]))

            test_log_loss += self.quick_train(X, y, epochs, learning_rate, l2)

        test_log_loss /= k_fold

        print(
            f"Epochs: {epochs} {'Learning Rate:':>20} {learning_rate} {'R2 Lambda:':>15} {l2:<6} {'Test Log Loss:':>25} {test_log_loss}"
        )

    def predict(self, X: list[list]) -> list[int]:
        y_probability = self.forward_propagation(X)

        # transform predicted probabilities into predicted class labels
        y_predict = [0 if prob < 0.5 else 1 for prob in y_probability]

        return y_probability, y_predict

    def evaluate(self, X: list[list], y: list[int]) -> tuple[float, float]:
        y_probability, y_predict = self.predict(X)
        accuracy = metrics.accuracy_score(y, y_predict)
        log_loss = metrics.log_loss(y, y_probability)
        return accuracy, log_loss

    def print_result(self, epoch, train_accuracy, train_log_loss, test_accuracy, test_log_loss, *args):
        print(
            f"Epoch {epoch:>3}: {'Train accuracy':>18} {train_accuracy:8.4f} {'Train loss:':>15} {train_log_loss:8.4f} {'Test accuracy':>18} {test_accuracy:8.4f} {'Test loss:':>12} {test_log_loss:2.4f} {'Weights:':>15}",
            end=" ",
        )
        for arg in args:
            print(arg, end=" ")

        print()

    def draw_result(self, scores):
        scores = np.array(scores)

        plt.subplot(1, 2, 1)
        plt.title("Accuracy")
        plt.plot(scores[:, 0], label="Train", color="tab:blue")
        plt.plot(scores[:, 2], label="Test", color="tab:green")
        plt.xlabel("Epoch")
        plt.legend(loc="lower right")

        plt.subplot(1, 2, 2)
        plt.title("Log Loss")
        plt.plot(scores[:, 1], label="Train", color="tab:blue")
        plt.plot(scores[:, 3], label="Test", color="tab:green")
        plt.xlabel("Epoch")
        plt.legend(loc="lower left")
        plt.show()


# Neural Network with 1 hidden layer
class NN2(NN):
    def __init__(self, n_hidden_nodes) -> None:
        """
        The hidden layer contains n_hidden_nodes and one bias node
        """
        self.n_hidden_nodes = n_hidden_nodes

    def setup_weight(self):
        """
        W1: list[list]
            Matrix of weights between input and hidden layers
        w2: list[float]
            List of weights between hidden and output layers
        """
        self.W1 = np.random.rand(len(self.X_train[0]), self.n_hidden_nodes)
        self.w2 = np.random.rand(self.n_hidden_nodes + 1)

        self.original_W1 = np.array(self.W1)
        self.original_w2 = np.array(self.w2)

    def reset_weight(self):
        """
        This function is called before each training in cross validation to make comparison between different parameters fairer.
        """
        self.W1 = np.array(self.original_W1)
        self.w2 = np.array(self.original_w2)

    def leaky_relu(self, X: list[float] | list[list], w: list[list]) -> list[float] | list[list]:
        """
        Calculate leaky RELU function: f(z) = z if z >= 0 and f(z) = 0.01*z if z < 0.

        If X is an 1-d array (representing a single input vector),  return a list representing a hidden vector.
        If X is a  2-d array (representing multiple input vectors), return a list of list representing a list of hidden vectors.
        """
        z = X @ w
        for n in np.nditer(z, op_flags=["readwrite"]):
            if n < 0:
                n[...] *= 0.01
        return z

    def forward_propagation(self, X: list[float] | list[list]) -> float | list[float]:
        """
        If X is an 1-d array (representing a single input vector),  return a single value representing the final output.
        If X is a  2-d array (representing multiple input vectors), return a list of floats representing a list of corresponding final outputs.
        """
        self.h_1dim = np.ones(5)
        self.h_2dim = np.ones((len(X), 5))

        if X.ndim == 1:
            # hidden layer
            self.h_1dim[1:] = self.leaky_relu(X, self.W1)
            # output layer
            return self.sigmoid(self.h_1dim, self.w2)
        else:
            # hidden layer
            self.h_2dim[:, 1:] = self.leaky_relu(X, self.W1)
            # output layer
            return self.sigmoid(self.h_2dim, self.w2)

    def backward_propagation(
        self, X: list[float] | list[list], y: int | list[int], l2_1: float, l2_2: float
    ) -> (list[float], list[list]):
        """
        Parameters
        ----------
        X: list | list[list]
            A single input vector | Multiple input vectors
        y: int | list
            A single target value | A list of target values
        l2_1: float
            Lambda factor in L2 regularization of the weights between input and hidden layers
        l2_2: float
            Lambda factor in L2 regularization of the weights between hidden and output layers

        Return
        ------
        dw2: list[float]
            List of the gradients of the weights between hidden and output layers
        DW1: list[list]
            Matrix of the gradients of the weights between input and hidden layers
        """
        y_predict = self.forward_propagation(X)

        if X.ndim == 1:
            dw2 = (y_predict - y) * self.h_1dim - l2_2 * self.w2

            # derivative of leaky RELU
            dh_dz = np.array([0.01 if z < 0 else 1 for z in self.h_1dim[1:]])
            DW1 = (y_predict - y) * (np.reshape(X, (5, 1)) @ np.reshape(self.w2[1:] * dh_dz, (1, 4))) - l2_1 * self.W1

        else:
            dw2 = self.h_2dim.T @ (y_predict - y) / len(X) - l2_2 * self.w2

            DH_DZ = self.h_2dim[:, 1:]
            for n in np.nditer(DH_DZ, op_flags=["readwrite"]):
                if n < 0:
                    n[...] = 0.01
                else:
                    n[...] = 1

            DW1 = (
                X.T @ (np.reshape((y_predict - y), (len(X), 1)) @ np.reshape(self.w2[1:], (1, 4)) * DH_DZ) / len(X)
                - l2_1 * self.W1
            )

        return dw2, DW1

    def quick_train(
        self,
        X: list[list],
        y: list[int],
        epochs: int = 1,
        learning_rate_1: float = 1,
        learning_rate_2: float = 1,
        l2_1: float = 0,
        l2_2: float = 0,
        print_result: bool = False,
    ) -> float:
        """
        Train the neural network and update weights after the forward pass of each input vector.

        Parameters
        ----------
        X: list[list]
            Training data input
        y: list[int]
            Training data target
        epochs: int
            Number of training iterations
        l2_1: float
            Lambda factor in L2 regularization of the weights between input and hidden layers
        l2_2: float
            Lambda factor in L2 regularization of the weights between hidden and output layers
        print_result: bool
            Whether to print some result after each training iteration

        Return
        -------
        test_log_loss: float
            Log loss on test data at the end of the training
        """
        scores = []
        self.reset_weight()
        for epoch in range(1, epochs + 1):
            for i in range(len(X)):
                dw2, DW1 = self.backward_propagation(X[i], y[i], l2_1, l2_2)
                self.w2 += learning_rate_2 * dw2
                self.W1 += learning_rate_1 * DW1

            train_accuracy, train_log_loss = self.evaluate(X, y)
            test_accuracy, test_log_loss = self.evaluate(self.X_test, self.y_test)

            if print_result:
                self.print_result(
                    epoch, train_accuracy, train_log_loss, test_accuracy, test_log_loss, self.W1[0], self.w2
                )
                scores.append([train_accuracy, train_log_loss, test_accuracy, test_log_loss])

        if print_result:
            self.draw_result(scores)

        return test_log_loss

    def slow_train(
        self, X, y, epochs=1, learning_rate_1=1, learning_rate_2=1, l2_1=0, l2_2=0, print_result=False
    ) -> float:
        """
        Train the neural network and update weights after each entire epoch forward pass.

        Parameters & Return
        -------------------
        Same as method quick_train()
        """
        scores = []
        self.reset_weight()
        for epoch in range(1, epochs + 1):
            dw2, DW1 = self.backward_propagation(X, y, l2_1, l2_2)
            self.w2 += learning_rate_2 * dw2
            self.W1 += learning_rate_1 * DW1

            train_accuracy, train_log_loss = self.evaluate(X, y)
            test_accuracy, test_log_loss = self.evaluate(self.X_test, self.y_test)

            if print_result:
                self.print_result(
                    epoch, train_accuracy, train_log_loss, test_accuracy, test_log_loss, self.W1[0], self.w2
                )
                scores.append([train_accuracy, train_log_loss, test_accuracy, test_log_loss])

        if print_result:
            self.draw_result(scores)

        return test_log_loss

    def cross_validation(
        self, k_fold: int, epochs: int, learning_rate_1: float, learning_rate_2: float, l2_1: float, l2_2: float
    ) -> None:
        """
        Train the neural network with cross validation and report the average log loss.
        """
        test_log_loss = 0
        for i in range(k_fold):
            n = len(self.X_train) // k_fold

            X = np.vstack((self.X_train[: i * n], self.X_train[(i + 1) * n :]))
            y = np.hstack((self.y_train[: i * n], self.y_train[(i + 1) * n :]))

            test_log_loss += self.quick_train(X, y, epochs, learning_rate_1, learning_rate_2, l2_1, l2_2)

        test_log_loss /= k_fold

        print(
            f"Epochs: {epochs} {'Hidden Layer Learning Rate:':>30} {learning_rate_1} {'Output Layer Learning Rate:':>30} {learning_rate_2} {'Hidden Layer R2 Lambda:':>30} {l2_1:<6} {'Output Layer R2 Lambda:':>30} {l2_2:<6} {'Test Log Loss:':>30} {test_log_loss}"
        )


X_train, X_test, y_train, y_test = read_data("data_banknote_authentication.txt", 0.2)
sklearn_logistic_reg(X_train, X_test, y_train, y_test)


nn = NN()
nn.import_data(X_train, X_test, y_train, y_test, normalization=True)
# nn.quick_train(nn.X_train, nn.y_train, epochs=100, learning_rate=0.2, l2=0.00, print_result=True)
# nn.slow_train(nn.X_train, nn.y_train, epochs=100, learning_rate=0.2, l2=0.00, print_result=True)
# nn.slow_train(nn.X_train, nn.y_train, epochs=100, learning_rate=1, l2=0.00, print_result=True)
# nn.slow_train(nn.X_train, nn.y_train, epochs=100, learning_rate=5, l2=0.00, print_result=True)
# nn.cross_validation(k_fold=10, epochs=20, learning_rate=0.2, l2=0)


nn2 = NN2(4)
nn2.import_data(X_train, X_test, y_train, y_test, normalization=True)
# nn2.quick_train(
#     nn2.X_train,
#     nn2.y_train,
#     epochs=100,
#     learning_rate_1=0.2,
#     learning_rate_2=0.2,
#     l2_1=0.0,
#     l2_2=0.0,
#     print_result=True,
# )
# nn2.quick_train(
#     nn2.X_train,
#     nn2.y_train,
#     epochs=100,
#     learning_rate_1=0.2,
#     learning_rate_2=0.2,
#     l2_1=0.001,
#     l2_2=0.001,
#     print_result=True,
# )
nn2.quick_train(
    nn2.X_train,
    nn2.y_train,
    epochs=100,
    learning_rate_1=0.2,
    learning_rate_2=0.2,
    l2_1=0.01,
    l2_2=0.01,
    print_result=True,
)
# nn2.slow_train(
#     nn2.X_train,
#     nn2.y_train,
#     epochs=100,
#     learning_rate_1=0.2,
#     learning_rate_2=0.2,
#     l2_1=0.0,
#     l2_2=0.0,
#     print_result=True,
# )
# nn2.slow_train(
#     nn2.X_train,
#     nn2.y_train,
#     epochs=100,
#     learning_rate_1=1,
#     learning_rate_2=1,
#     l2_1=0.0,
#     l2_2=0.0,
#     print_result=True,
# )
# nn2.slow_train(
#     nn2.X_train,
#     nn2.y_train,
#     epochs=100,
#     learning_rate_1=5,
#     learning_rate_2=5,
#     l2_1=0.0,
#     l2_2=0.0,
#     print_result=True,
# )

# Tuning l2
# l2s = [0.001, 0.005, 0.01, 0.1]
# for l2 in l2s:
#     nn2.cross_validation(k_fold=10, epochs=20, learning_rate_1=0.2, learning_rate_2=0.2, l2_1=l2, l2_2=l2)

# nn2.cross_validation(k_fold=10, epochs=20, learning_rate_1=0.2, learning_rate_2=0.2, l2_1=0, l2_2=0.01)
# nn2.cross_validation(k_fold=10, epochs=20, learning_rate_1=0.2, learning_rate_2=0.2, l2_1=0.001, l2_2=0.01)
# nn2.cross_validation(k_fold=10, epochs=20, learning_rate_1=0.2, learning_rate_2=0.2, l2_1=0.002, l2_2=0.01)
# nn2.cross_validation(k_fold=10, epochs=20, learning_rate_1=0.2, learning_rate_2=0.2, l2_1=0.005, l2_2=0.01)
# nn2.cross_validation(k_fold=10, epochs=20, learning_rate_1=0.2, learning_rate_2=0.2, l2_1=0.01, l2_2=0.01)
# nn2.cross_validation(k_fold=10, epochs=20, learning_rate_1=0.2, learning_rate_2=0.2, l2_1=0.02, l2_2=0.01)
# nn2.cross_validation(k_fold=10, epochs=20, learning_rate_1=0.2, learning_rate_2=0.2, l2_1=0.05, l2_2=0.01)

# nn2.cross_validation(k_fold=10, epochs=20, learning_rate_1=0.2, learning_rate_2=0.2, l2_1=0.001, l2_2=0)
# nn2.cross_validation(k_fold=10, epochs=20, learning_rate_1=0.2, learning_rate_2=0.2, l2_1=0.001, l2_2=0.001)
# nn2.cross_validation(k_fold=10, epochs=20, learning_rate_1=0.2, learning_rate_2=0.2, l2_1=0.001, l2_2=0.002)
# nn2.cross_validation(k_fold=10, epochs=20, learning_rate_1=0.2, learning_rate_2=0.2, l2_1=0.001, l2_2=0.005)
# nn2.cross_validation(k_fold=10, epochs=20, learning_rate_1=0.2, learning_rate_2=0.2, l2_1=0.001, l2_2=0.01)
# nn2.cross_validation(k_fold=10, epochs=20, learning_rate_1=0.2, learning_rate_2=0.2, l2_1=0.001, l2_2=0.02)
# nn2.cross_validation(k_fold=10, epochs=20, learning_rate_1=0.2, learning_rate_2=0.2, l2_1=0.001, l2_2=0.05)

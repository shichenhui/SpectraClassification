import numpy as np
import MyUtils


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


class ELM:
    def __init__(self, num_hidden):
        self.num_hidden = num_hidden
        self.w1 = None
        self.b = None
        self.beta1 = None
        pass

    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            pass
        else:
            X = np.array(X)
        if isinstance(y, np.ndarray):
            pass
        else:
            y = np.array(y)
        # -------------Auto Encoder----------------------
        self.w1 = np.random.uniform(-1, 1, size=(X.shape[1], self.num_hidden))
        # self.b = np.zeros(shape=(self.num_hidden,))
        self.b1 = np.random.uniform(-1, 1, size=(self.num_hidden))

        h = sigmoid(X.dot(self.w1) + self.b1)
        self.h_pinv = np.linalg.pinv(h)
        # self.w2 = self.h_pinv.dot(y)
        self.beta1 = self.h_pinv.dot(X)

        # --------------ELM------------------
        feature = sigmoid(X.dot(self.w1) + self.b1)
        self.w2 = np.random.uniform(-1, 1, size=(feature.shape[1], self.num_hidden))
        self.b2 = np.random.uniform(-1, 1, size=(self.num_hidden))
        h2 = sigmoid(feature.dot(self.w2) + self.b2)
        h_pinv2 = np.linalg.pinv(h2)
        self.beta2 = h_pinv2.dot(y)

    def predict(self, X):
        y_pred = self.h_pinv.dot(self.w2)
        return y_pred

    def evaluate(self, X, y):
        feature = sigmoid(X.dot(self.w1) + self.b1)
        y_pred = sigmoid(feature.dot(self.w2) + self.b2).dot(self.beta2)
        accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)) / len(y)
        print(accuracy)


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = MyUtils.load_data(r'C:\Users\panda\Desktop\光谱数据样例\star_AFGK_2kx4.csv',
                                                         class_num=4,
                                                         norm=True, shuffle=True, split=0.8, one_hot=True)

    model = ELM(300)
    model.fit(X_train, y_train)
    model.evaluate(X_test, y_test)

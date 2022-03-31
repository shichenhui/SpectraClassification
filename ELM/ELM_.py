import numpy as np
import MyUtils


def sigmoid(x):
    return 1. / (1. + np.exp(-x))

class ELM:
    def __init__(self, num_hidden):
        self.num_hidden = num_hidden
        self.w1 = None
        self.b = None
        self.w2 = None
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

        self.w1 = np.random.uniform(-1, 1, size=(X.shape[1], self.num_hidden))
        #self.b = np.zeros(shape=(self.num_hidden,))
        self.b = np.random.uniform(-1, 1, size=(self.num_hidden))

        h = sigmoid(X.dot(self.w1) + self.b)
        self.h_pinv = np.linalg.pinv(h)
        self.w2 = self.h_pinv.dot(y)
        #self.w2 = np.linalg.inv(np.transpose(h) * h) * np.transpose(h) * np.transpose(y)

    def predict(self, X):
        y_pred = self.h_pinv.dot(self.w2)
        return y_pred

    def evaluate(self, X, y):
        y_pred = sigmoid(X.dot(self.w1)+self.b).dot(self.w2)
        accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)) / len(y)
        print(accuracy)
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = MyUtils.load_data(r'E:\spectra_ML\光谱数据样例\star_AFGK_2kx4.csv', class_num=4,
                                                     norm=True, shuffle=True, split=0.8, one_hot=True)

    model = ELM(300)
    model.fit(X_train, y_train)
    model.evaluate(X_test, y_test)

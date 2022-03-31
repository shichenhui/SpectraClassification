import numpy as np
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA


def load_data(path: str, class_num, norm=False, shuffle=False, one_hot=False, split=None):
    """
    Load dataset
    :param norm:
    :param path: str
    :param class_num: Number of each class, int or iterable,
        if class_num = n, indicate that the dataset n equal classes
    :param shuffle: Shuffle or not.
    :param one_hot: Translate label to one_hot encode or not.
    :param split: split dataset into train and test. If None, won't split dataset.
    :return: tuple. (dataset, label) or (X_train, y_train, X_test, y_test) depend on split para.
    """
    print('load data')
    data = np.loadtxt(path, delimiter=',')
    # Generate label
    label = []
    if isinstance(class_num, int):
        per = int(data.shape[0] / class_num)
        for i in range(class_num):
            label += [i] * per
        n = class_num
    else:
        for e, i in enumerate(class_num):
            label += [e] * i
        n = len(class_num)
    label = np.array(label)
    if shuffle:
        index_shuffle = np.random.permutation(data.shape[0])
        data = data[index_shuffle]
        label = label[index_shuffle]

    if one_hot:
        ohx = np.zeros((len(label), n))
        ohx[range(len(label)), label] = 1
        label = ohx

    if norm:
        data = normalize(data)

    if split:
        num_train = int(data.shape[0] * split)
        print(num_train)
        return data[:num_train], label[:num_train], data[num_train:], label[num_train:]

    return data, label


X_train, y_train, X_test, y_test = load_data(r'C:\Users\panda\Desktop\光谱数据样例\index_AFGK_2kx4.csv', class_num=4,
                                             norm=True, shuffle=True, split=0.7)
# pca = PCA(n_components=200)
# data = pca.fit_transform(data)

rf = RandomForestClassifier(n_estimators=100, verbose=True, n_jobs=-1)
rf.fit(X_train, y_train)

score = rf.score(X_test, y_test)
print(score)

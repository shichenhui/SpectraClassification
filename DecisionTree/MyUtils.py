import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import itertools

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

def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues, position=111):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    cm = np.array(cm)
    # f = plt.figure()
    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm[::-1]  # 第一个轴逆序
        print("Normalized confusion matrix")
    else:
        cm = cm[::-1]
        print('Confusion matrix, without normalization')
    print(cm)
    plt.subplot(position)
    plt.imshow(cm, interpolation='nearest', cmap=cmap, norm=None)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(cm))
    xtick_marks = np.arange(len(cm[1]))
    print(classes, len(cm[1]))
    plt.xticks(xtick_marks, classes[:len(cm[1])])
    plt.yticks(tick_marks, classes[:len(cm)][::-1])
    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

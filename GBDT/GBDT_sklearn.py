import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn.model_selection import cross_val_score, GridSearchCV



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
                                                     norm=True, shuffle=True, split=0.8)

# param_test1 = {'min_samples_leaf':range(20,101,10)}
# gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(n_estimators=70, learning_rate=0.1, min_samples_split=200,
#                                   min_samples_leaf=90,max_depth=13,max_features=None, subsample=0.8,random_state=10), 
#                        param_grid = param_test1, scoring='f1_micro',cv=5,n_jobs=-1)
# gsearch1.fit(X_train, y_train)
# print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)

model1 = GradientBoostingClassifier(n_estimators=70, learning_rate=0.1, min_samples_split=3,
                                   min_samples_leaf=2,max_depth=13,max_features=None, subsample=0.8,random_state=10)

model = GradientBoostingClassifier(random_state=10)

score = cross_val_score(model, X_train, y_train,cv=5,n_jobs=-1,scoring='f1_micro')
score1 = cross_val_score(model1, X_train, y_train,cv=5,n_jobs=-1,scoring='f1_micro')

print(score)
print(score.mean())
print(score1)
print(score1.mean())

import numpy as np
from sklearn import svm
import sys
import MyUtils
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import itertools

# sys.path.append("..")

X_train, y_train = MyUtils.load_data(r'C:\Users\panda\Desktop\光谱数据样例\para_AFGK_2kx4.csv', class_num=4,
                                                     norm=0, shuffle=True, split=0)

model = svm.SVC(C=5, verbose=0, kernel='rbf', decision_function_shape='ovo')

#model.fit(X_train, y_train)

paras = {'C':[61]}
gsearch1 = GridSearchCV(estimator = model,
                       param_grid = paras, scoring='f1_micro',cv=2,n_jobs=-1)
gsearch1.fit(X_train, y_train)
print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)

# score = model.score(X_test, y_test)
# y_predict = model.predict(X_test)
# print(y_predict)
# print(np.sum(y_test == 0))
# cm = confusion_matrix(y_test, y_predict)
# MyUtils.plot_confusion_matrix(cm, ['A', 'F', 'G', 'K'], normalize=1)
# print('\n\n', score)

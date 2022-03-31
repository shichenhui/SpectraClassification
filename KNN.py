from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import MyUtils

X_train, y_train = MyUtils.load_data(r'/home/shichenhui/code/spectra_clustering/data/-10/star_AFGK_2kx4.csv',
                                     class_num=4,
                                     norm=True, shuffle=True, split=0)

model = KNeighborsClassifier(n_neighbors=2)

paras = {'n_neighbors': range(2, 50, 2)}
gsearch1 = GridSearchCV(estimator=model,
                        param_grid=paras, scoring='f1_micro', cv=2, n_jobs=-1)
gsearch1.fit(X_train, y_train)
print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)

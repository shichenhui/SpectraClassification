from sklearn import tree
import MyUtils
from sklearn.model_selection import cross_val_score, cross_val_predict

model = tree.DecisionTreeClassifier(criterion='entropy')

# X_train, y_train, X_test, y_test = MyUtils.load_data(r'C:\Users\panda\Desktop\光谱数据样例\star_AFGK_2kx4.csv', class_num=4,
#                                                      norm=True, shuffle=True, split=0.7)
# model.fit(X_train, y_train)

# score = model.score(X_test, y_test)
# print(score)
X_train, y_train = MyUtils.load_data(r'C:\Users\panda\Desktop\光谱数据样例\index_AFGK_2kx4.csv', class_num=4,
                                     norm=True, shuffle=True, split=None)

score = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1, scoring='f1_micro')
print(score.mean())

import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = [0, 0, 1, 1]

clf = LogisticRegression(random_state=0).fit(X, y)
print(clf.score(X, y))
print(clf.coef_)
print(clf.intercept_)

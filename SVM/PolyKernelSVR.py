import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel()
# add noise to targets
y[::5] += 2 * (0.5 - np.random.rand(20))
svr = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1,
coef0=1)
ax = plt.gca()
#plot regression curve
ax.plot(X,svr.fit(X, y).predict(X),color="k",lw=1)
ax.scatter(X[svr.support_],y[svr.support_],facecolor="none",
edgecolor="k",s=20)
plt.show()
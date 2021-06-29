from pandas import np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import random

m = 20
X1, Y1 = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, n_classes=2, n_samples=m, class_sep = 10)
X_train = np.hstack((X1, np.ones((X1.shape[0], 1), dtype=X1.dtype)))

def conversion(Y):
    for i in range(Y.shape[0]):
        if Y[i] == 0:
            Y[i] = -1

def classification(E, Y_Blad, X, Y, w):
    for i in range(len(X)):
        s = w[0] + w[1] * X[i,0] + w[2] * X[i,1]
        f = s > 0 and 1 or -1
        if f != Y[i]:
            E.append(X[i])
            Y_Blad.append(Y[i])

def perceptron(X, Y):
    #generating the necessary initial structures

    w = np.zeros(3) #from 1 xi1 xi2
    k = 0 #number of plugs
    eta = 0.3 # from 0 to 1
    E = []
    Y_Blad = []

    #first classification

    classification(E, Y_Blad, X, Y, w)

    while(len(E) > 0):

        #accutalization of scales

        losowy_Indeks = random.randrange(len(E))
        for i in range(3):
            w[i] = w[i] + eta * Y_Blad[losowy_Indeks] * E[losowy_Indeks][i]

        #Resetting the E-list

        E = []
        Y_Blad = []

        #another classification with new weights

        classification(E, Y_Blad, X, Y, w)

        k += 1
        
        #Condition if the algorithm would run too long 
        
        if k >= 1000: 
            break
    return w, k

conversion(Y1)
w, k = perceptron(X_train, Y1)

plt.scatter(X1[:, 0], X1[:, 1], marker='x', c=Y1, s=25, edgecolor='k')

x = np.arange(10)
y = np.zeros(10)
for i in range(10):
    y[i] = (-(w[0]/w[1]) * x[i] - w[2]/w[1])
plt.plot(x,y)
plt.show()
print("w: ", w)
print("algorithm execution plugs: ", k)
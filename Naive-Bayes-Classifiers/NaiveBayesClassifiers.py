import numpy as np
from sklearn.base import  BaseEstimator, ClassifierMixin
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

class NBC (BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.Papriori = None # vector with a priori probabilities of classes
        self.labels_classes = None # wine labels
        self.number_of_appearances = []
        self.probabilities = []
        self.result_predict = []
        self.result_predict_test = []

    def fit(self, X_T, Y_T, bin, switch):

        m, n = X_T.shape #rows columns
        self.labels_classes = np.unique(Y_T) # 1, 2, 3
        self.Papriori = np.zeros(3)
        print("Labels: " + str(self.labels_classes))
        Y_TMP = np.zeros(len(Y_T))
        #P(Y=y)
        for i in range(len(Y_T)): #'i' consecutive numbers and the 'label' is the current
            if(Y_T[i] == self.labels_classes[0]):
                self.Papriori[0] += 1
            elif (Y_T[i] == self.labels_classes[1]):
                self.Papriori[1] += 1
            elif (Y_T[i] == self.labels_classes[2]):
                self.Papriori[2] += 1
        for i in range(len(self.Papriori)):
            if switch == 0:
                self.Papriori[i] = self.Papriori[i] / len(Y_T)
            elif switch == 1:
                self.Papriori[i] = (self.Papriori[i] + 1) / (len(Y_T) + np.size(np.unique(X_T)))

        print("Apriori 0: " + str(self.Papriori[0]))
        print("Apriori 1: " + str(self.Papriori[1]))
        print("Apriori 2: " + str(self.Papriori[2]))

        self.number_of_appearances = np.zeros((3,n,bin)) #classes, features, bins
        self.probabilities = np.zeros((3,n,bin))

        #P(X|Y)
        for i in range(len(Y_T)):
            for j in range(n):
                for k in range(len(self.labels_classes)):
                    if (Y_T[i] == self.labels_classes[k]):
                        self.number_of_appearances[k, j, int(X_T[i, j])] += 1

        for i in range(bin):
            for j in range(n):
                for k in range(len(self.labels_classes)):
                    if switch == 0:
                        self.probabilities[k, j, i] = self.number_of_appearances[k, j, i] / sum(self.number_of_appearances[k, j])
                    elif switch == 1:
                        self.probabilities[k, j, i] = (self.number_of_appearances[k, j, i] + 1) / (sum(self.number_of_appearances[k, j]) + np.size(np.unique(X_T)))


        print("Number of appearances: " + str(self.number_of_appearances))
        print("Probabilities for features: ")
        print(self.probabilities)

    def predict(self, X_T):
        m, n = X_T.shape # rows columns
        for k in range(m):
            tmp = np.ones((len(self.labels_classes),1))
            for i in range(len(self.labels_classes)):
                for j in range(n):
                    tmp[i] = tmp[i] * self.Papriori[i] * self.probabilities[i,j,int(X_T[k,j])]
            self.result_predict.append(np.argmax(tmp) + 1)
        return self.result_predict

    def predict_test(self, X_T):
        lista_tmp = []
        m, n = X_T.shape  # rows columns
        for k in range(m):
            tmp = np.ones((len(self.labels_classes), 1))
            for i in range(len(self.labels_classes)):
                for j in range(n):
                    tmp[i] = tmp[i] * self.Papriori[i] * self.probabilities[i, j, int(X_T[k, j])]
            lista_tmp.append(tmp[i] / sum(tmp[i]))
            self.result_predict_test.append(np.argmax(lista_tmp) + 1)
        return self.result_predict_test

    def accuracy(self, Y_True, Y_Test, size):
        counter = 0
        for i in range(size):
            if Y_True[i] == Y_Test[i]:
                counter += 1
        return ((counter / size) * 100)

if __name__== '__main__':
    Gausian = GaussianNB()
    
    Data = np.genfromtxt("wine.data" , delimiter = ",")
    Y = Data[:,0]
    bin = 2
    X = KBinsDiscretizer(bin, 'ordinal', 'uniform')
    X = X.fit_transform(Data[:, 1:])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    
    Gausian.fit(X, Y)
    list_Gausian = Gausian.predict(X_test)
    
    print("X_train")
    print("Quantity: " + str(len(X_train)))
    print(X_train)
    print("------------------------")
    print("Y_train")
    print("Quantity: " + str(len(Y_train)))
    print(Y_train)
    print("------------------------")
    N = NBC()

    N.fit(X_train, Y_train, bin, 1)
    my_list_NBC = N.predict(X_test)
    my_list_NBC_test = N.predict_test(X_test)

    print("My list: ", my_list_NBC)
    print("---------------------")
    print("Gausian list: ", list_Gausian)
    print("---------------------")
    print("Y_test: ", Y_test)

    print("Gausian: ", N.accuracy(list_Gausian, my_list_NBC, np.size(my_list_NBC)))
    print("Accuracy: ",  N.accuracy(my_list_NBC, Y_test, np.size(my_list_NBC_test)))

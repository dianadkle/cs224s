from sklearn.neighbors import KNeighborsClassifier
X = [[0], [1], [2], [3]] # training data
y = [0, 0, 1, 1] # labels

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y) 

labels = neigh.predict([[1.1]]) # put in matrix to be predicted on
print labels 
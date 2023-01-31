from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
import numpy as np

X,Y = load_iris().data, load_iris().target

mlp = MLPClassifier()
mlp.fit(X, Y)

a = mlp.predict([3.1,  2.5,  8.4,  2.2])
b = mlp.predict_proba([3.1,  2.5,  8.4,  2.2])
print("sum: %f"%np.sum(mlp.predict_proba([3.1,  2.5,  8.4,  2.2])))
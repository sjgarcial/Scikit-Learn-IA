from sklearn.datasets import load_iris # database de colores mas utilizado
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# cargar el conjunto de datos iris

iris=load_iris()
X_train, X_test, y_train, y_test= train_test_split(
    iris.data,iris.target,test_size=0.2,random_state=42)

# crear el clasificador de vecinos mas cercanos

clf=KNeighborsClassifier(n_neighbors=3)

# Entrenar el clasificador
clf.fit(X_train, y_train)

#  predecir las etiquetas para los datos de prueba

y_pred=clf.predict(X_test)

# calcular la precision del clasificador

accuracy=accuracy_score(y_test, y_pred)
print('Precision del clasificador:', accuracy)


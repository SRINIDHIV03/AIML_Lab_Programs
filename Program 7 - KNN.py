import numpy as np
from collections import Counter

class KNN:
    def __init__(self,k=3):
        self.k=k
    
    def fit(self,X,y):
        self.X_train=X
        self.y_train=y
    
    def predict(self,X):
        y_pred=[self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self,x):
        distances=[np.linalg.norm(x-x_train) for x_train in self.X_train]
        k_indices=np.argsort(distances)[:self.k]
        k_nearest_labels=[self.y_train[i] for i in k_indices]
        most_common=Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

iris=load_breast_cancer()
X,y=iris.data,iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
knn=KNN()
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)

print("Accuracy: ",accuracy_score(y_test,y_pred))
print("Classification Report:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))

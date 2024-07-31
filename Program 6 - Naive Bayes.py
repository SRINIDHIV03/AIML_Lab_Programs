import numpy as np

class NaiveBayes:

    def fit(self,X,y):
        n_samples,n_features=X.shape
        self._classes=np.unique(y)
        n_classes=len(self._classes)

        self._mean=np.zeros((n_classes,n_features),dtype=np.float64)
        self._var=np.zeros((n_classes,n_features),dtype=np.float64)
        self._prior=np.zeros(n_classes,dtype=np.float64)

        for idx,c in enumerate(self._classes):
            X_c=X[y==c]
            self._mean[idx,:]=X_c.mean(axis=0)
            self._var[idx,:]=X_c.var(axis=0)
            self._prior[idx]=X_c.shape[0]/float(n_samples)
        
    def predict(self,X):
        y_pred=[self._predict(x) for x in X]
        return np.array(y_pred)
        
    def _predict(self,x):
        posteriors=[]
        for idx,c in enumerate(self._classes):
            prior=np.log(self._prior[idx])
            posterior=np.sum(np.log(self._pdf(idx,x)))
            posterior=posterior+prior
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]
    
    def _pdf(self,idx,x):
        mean=self._mean[idx]
        var=self._var[idx]
        numerator=np.exp(-((x-mean)**2)/(2*var))
        denominator=np.sqrt(2*np.pi*var)
        return numerator/denominator
    
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

iris=load_iris()
X,y=iris.data,iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
nb=NaiveBayes()
nb.fit(X_train,y_train)

y_pred=nb.predict(X_test)

print("Accuracy: ",accuracy_score(y_test,y_pred))
print("Classification Report:\n",classification_report(y_pred,y_test))
print("Confusion Matrix:\n",confusion_matrix(y_pred,y_test))
        


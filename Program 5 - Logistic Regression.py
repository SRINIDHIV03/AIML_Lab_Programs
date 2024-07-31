import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression:
    def __init__(self,lr=0.001,n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.weights=None
        self.bias=None
    def fit(self,X,y):
        n_samples,n_features=X.shape
        self.weights=np.zeros(n_features)
        self.bias=0

        for _ in range(self.n_iters):
            linear_pred=np.dot(X,self.weights)+self.bias
            class_pred=sigmoid(linear_pred)

            dw=(1/n_samples)*np.dot(X.T,(class_pred-y))
            db=(1/n_samples)*np.sum(class_pred-y)

            self.weights=self.weights-self.lr*dw
            self.bias=self.bias-self.lr*db

    def predict(self,X):
        linear_pred=np.dot(X, self.weights)+ self.bias
        class_pred=sigmoid(linear_pred)
        y_pred=[0 if y<=0.5 else 1 for y in class_pred]
        return y_pred
    
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler

iris=datasets.load_iris()
X,y=iris.data,iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# scaler=StandardScaler()

# X_train=scaler.fit_transform(X_train)
# X_test=scaler.fit_transform(X_test)

lb=LogisticRegression()
lb.fit(X_train,y_train)
y_pred=lb.predict(X_test)

print("Accuracy: ",accuracy_score(y_test,y_pred))





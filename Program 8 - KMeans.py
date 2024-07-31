import numpy as np

class KMeans:
    def __init__(self,k=3,max_iters=1000):
        self.k=k
        self.max_iters=max_iters
        self.clusters=[[] for _ in range(self.k)]
        self.centroids=[]

    def fit(self,X):
        self.X=X
        self.n_samples,self.n_features=X.shape
        random_samples_idx=np.random.choice(self.n_samples,self.k,replace=False)
        self.centroids=[self.X[idx] for idx in random_samples_idx]

        for _ in range(self.max_iters):
            self.clusters=self._create_clusters(self.centroids)
            centroids_old=self.centroids
            self.centroids=self._get_centroids(self.clusters)
            if self._is_converged(self.centroids,centroids_old):
                break
    
    def _create_clusters(self,centroids):
        clusters=[[] for _ in range(self.k)]
        for idx,sample in enumerate(self.X):
            closest_idx=self._closest_centroid(sample,centroids)
            clusters[closest_idx].append(idx)
        return clusters
    
    def _closest_centroid(self,sample,centroids):
        distances=np.zeros(self.k)
        for idx,centroid in enumerate(centroids):
            distances[idx]=np.linalg.norm(sample-centroid)
        return np.argmin(distances)
    
    def _get_centroids(self,clusters):
        centroids=np.zeros((self.k,self.n_features))
        for cluster_idx,cluster in enumerate(clusters):
            cluster_mean=np.mean(self.X[cluster],axis=0)
            centroids[cluster_idx]=cluster_mean
        return centroids
    
    def predict(self,X):
        labels=np.empty(self.n_samples)
        for idx,sample in enumerate(X):
            for cluster_idx,cluster in enumerate(self.clusters):
                if idx in cluster:
                    labels[idx]=cluster_idx
        return labels
    
    def _is_converged(self,centroids,centroids_old):
        disatnces=np.zeros(self.k)
        for idx in range(self.k):
            disatnces[idx]=np.linalg.norm(centroids[idx]-centroids_old[idx])
        return sum(disatnces)==0
    

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# Load iris dataset
iris = load_iris()
X = iris.data
class_names = iris.target_names
y=iris.target
# Split dataset into training set and test set
X_train, X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=1)


# Create a KMeans object
k = KMeans(k=3, max_iters=100)
k.fit(X_train)

# Predict the clusters for the data
y_pred = k.predict(X_train)
print(len(y_pred))
# Convert y_pred to int
y_pred = y_pred.astype(int)
# print(y_pred)
print("Predictions:", class_names[y_pred])

print(accuracy_score(y_train,y_pred))



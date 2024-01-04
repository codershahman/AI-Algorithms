############ Implement K-Means with numpy ############
import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.first_iter = True

    #Calculating the distance
    def calculate_distance(self, X, centroids):
        a = ((X-centroids[:,np.newaxis,:]))**2
        distances=(np.sum(a,axis=2)**0.5)
        distances=(np.transpose(distances))
        if self.first_iter:
            print(f"distances[0:5]:\n{distances[0:5]}")
        return distances

    #assigning labels according to distance
    def assign_labels(self, distances):
        # TODO 1.2
        # Assign each data point to the nearest centroid
        # Hint: You may consider using
            # np.argmin: https://numpy.org/doc/stable/reference/generated/numpy.argmin.html
        labels = np.argmin(distances,axis=1)

        if self.first_iter:
            print(f"labels[0:5]:\n{labels[0:150]}")
        return labels

    #updating the centroids according to the newly assigned labels
    def update_centroids(self, X, centroids, labels):


        for x in range(len(centroids)):
          positions=np.where(labels==x)
          centroids[x]=np.mean(X[positions],axis=0)
        if self.first_iter:
            print(f"centroids:\n{centroids}")
        return centroids

    #bringing it all together
    def fit(self, X):
        centroids = X[np.random.choice(range(len(X)), self.n_clusters, replace=False)]
        for i in range(self.max_iter):
            distances = self.calculate_distance(X, centroids)
            labels = self.assign_labels(distances)
            centroids = self.update_centroids(X, centroids, labels)
            self.first_iter = False
        print(f"final centroids:\n{centroids}")     # print the final centroids
        print(f'final labels:\n{labels}')           # print the final labels

        return labels, centroids

# run K-Means
np.random.seed(23333)
k = 3
my_kmeans = KMeans(n_clusters=k)
labels, centroids = my_kmeans.fit(X)

# plotting the result
let_me_see(X, k, labels, centroids)

import numpy as np
a = np.arange(9).reshape((3,3))
#print(a)
b = np.arange(105).reshape((5,7,3,1))
#print(b)
np.dot(a, b)


#Implementing sse evalution function for clustering result
# sum of squared errors (SSE)
def sse(X, labels, centroids):
    # TODO 2.1
    #print(labels)
    deltas = (np.arange(centroids.shape[0]) == labels.reshape(-1, 1))
    a=deltas[:,np.newaxis,:]*X[:,:,np.newaxis]
    b=(np.transpose(centroids)-a)
    c=(np.sum(((b*deltas[:,np.newaxis,:])**2),axis=1))
    sse=(np.sum(c))
    return sse

#Implementing accuracy evalution function for classification result ############
# accuracy (ACC)
def accuracy(y, y_pred):

    a=np.sum(((y==y_pred)*1))
    acc = a/y.shape[0]
    return acc

print(f'SSE: {sse(X, labels, centroids)}')

# aligning the labels with the ground truth
aligned_y_pred = align_labels(y, labels)

# calculating the accuracy
print(f'Accuracy: {accuracy(y, aligned_y_pred)}') 

# plotting the aligned labels
let_me_see(X, k, aligned_y_pred, centroids, True) 
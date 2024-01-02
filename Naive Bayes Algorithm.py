from google.colab import drive
drive.mount("/content/drive")

# %cd "/content/drive/MyDrive/Colab Notebooks/Lab2"
# %ls

import numpy as np
train_dataset = np.load("train_dataset.npy")
test_dataset = np.load("test_dataset.npy")
train_labels = np.load("train_labels.npy")
test_labels = np.load("test_labels.npy")

from numpy.core.multiarray import array
class NaiveBayesClassifier:
  def __init__(self):
    self.train_dataset = None
    self.train_labels = None
    self.train_size = 0
    self.num_features = 0
    self.num_classes = 0
    self.num_feature_categories = 0


#Fitting the training data
  def fit(self, train_dataset, train_labels):
    self.train_dataset = train_dataset
    self.train_labels = train_labels
    self.train_size = self.train_dataset.shape[0] #number of training data entries
    self.num_features = self.train_dataset.shape[1] #number of columns for each entry: 6
    self.num_classes =  np.amax(self.train_labels)+1 #number of possible results : 2
    self.num_feature_categories = np.amax(train_dataset,0)+1 #number of possible results for each feature: varies

#Estimating class prior probabilities
  def estimate_class_prior(self):
    deltas = (np.arange(self.num_classes) == self.train_labels.reshape(-1, 1))
    class_prior=(((np.sum(deltas,0))+1)/(self.train_size+self.num_classes))
    return class_prior

#Estimating likelihoods
  def estimate_likelihoods(self):
    likelihoods = []
    for feature in np.arange(self.num_features):
      feature_likelihood=np.empty((self.num_feature_categories[feature],self.num_classes))
      deltas = (np.arange(self.num_classes) == self.train_labels.reshape(-1, 1))
      deltas1=np.arange(self.num_feature_categories[feature])==self.train_dataset[:,feature].reshape(-1,1)
      for columns in np.arange((np.shape(deltas1)[1])):
        feature_likelihood[columns,:]=(((np.dot(deltas1[:,columns],(deltas)*1)+1)/(sum(deltas)+self.num_feature_categories[feature])))
      likelihoods.append(feature_likelihood)
    return likelihoods

#Finally, predicting the labels
  def predict(self, test_dataset):
    test_size = test_dataset.shape[0]
    class_prior = self.estimate_class_prior()
    likelihoods = self.estimate_likelihoods()
    class_prob = np.tile(np.log(class_prior), (test_size, 1))
    for feature in np.arange(self.num_features):
      feature_likelihood = likelihoods[feature]
      deltas2=np.arange(self.num_feature_categories[feature])==test_dataset[:,feature].reshape(-1,1)
      class_prob=(class_prob+ np.log(np.dot(deltas2,feature_likelihood)))
    test_predict=np.argmax(class_prob,1)
      #TO DO: Change the class_prob value based on the likelihood
    return test_predict
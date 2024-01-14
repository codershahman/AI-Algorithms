
#Cleaning text.  Remove punctuations, caps

if __name__ == '__main__':
  text = '''In mathematics, an open set is a generalization of an open interval in the real line.
  In a metric space (a set along with a distance defined between any two points),
  an open set is a set that, along with every point P, contains all points that are sufficiently near to P
  (that is, all points whose distance to P is less than some value depending on P).
  More generally, an open set is a member of a given collection of subsets of a given set,
  a collection that has the property of containing every union of its members, every finite intersection of its members,
  the empty set, and the whole set itself. A set in which such a collection is given is called a topological space,
  and the collection is called a topology. These conditions are very loose, and allow enormous flexibility in the choice of open sets.
  For example, every subset can be open (the discrete topology), or no subset can be open except the space itself and the empty set
  (the indiscrete topology). In practice, however, open sets are usually chosen to provide a notion of nearness that is
  similar to that of metric spaces, without having a notion of distance defined. In particular,
  a topology allows defining properties such as continuity, connectedness, and compactness, which were originally defined by means of a distance.'''


  import re  # python's regular expression library.  A powerful library for text manipulation
  cleaning = text.lower()  # CAPS -> caps
  cleaning = re.sub(r'\n', ' ', cleaning)  # replace the newline characters with a space
  cleaning = re.sub(r'[^a-z0-9 -]', '', cleaning)  # replace all characters except letters, numbers, space bars and hyphens with an empty string
  cleaned_text = re.sub(' +', ' ', cleaning)  # replace multiple spaces with one space

  print(cleaned_text)

#Basic text processing
def generate_variables(cleaned_text):

  words_in_order=(str.split(cleaned_text," "))
  num_vocabs=len(set(words_in_order))
  sorted1=sorted(set((words_in_order)))
  mylist1=zip(range(0,num_vocabs),sorted1)
  mylist2=zip(sorted1,range(0,num_vocabs))

  word_to_index={key:value for key,value in mylist2}
  index_to_word={key:value for key,value in mylist1}



  return (words_in_order,num_vocabs,word_to_index,index_to_word)

if __name__ == '__main__':
  words_in_order, num_vocabs, word_to_index, index_to_word = generate_variables(cleaned_text)

from numpy.core.multiarray import result_type
from numpy.lib.index_tricks import index_exp
#Generating training data
import numpy as np

#Implementing the cbow approach
def generate_training_data_cbow(words_in_order, word_to_index, window_size):

    mylist1=zip(range(0,len(words_in_order)),words_in_order)
    num_words=len((words_in_order))

    y_cbow=[word_to_index[word] for word in words_in_order]
    x_cbow=[[word_to_index[words_in_order[index-window_size+i]]if ((index-window_size+i)>-1 and (index-window_size+i)<num_words and (i!= window_size)) else -1 for i in range((2*window_size)+1)]for index,word in mylist1 ]
    x_cbow=[[element for element in list if element>-1]for list in x_cbow]

    for i in range(window_size):
      x_cbow=[list+[-1] if len(list)<2*window_size else list for list in x_cbow ]
      x_cbow=[list+[-1] if len(list)<2*window_size else list for list in x_cbow ]
    return (np.array(x_cbow),np.array(y_cbow))

#Implementing the skipgram approach
def generate_training_data_skipgram(words_in_order, word_to_index, window_size):
    num_words=len((words_in_order))
    mylist1=zip(range(0,len(words_in_order)),words_in_order)


    x_skipgram = [word_to_index[item] for index,item in mylist1 for i in range((2*window_size)+1) if ((-window_size+i)!=0 and index+(-window_size+i)>-1 and index+(-window_size+i)<num_words  )]
    y_skipgram = [word_to_index[words_in_order[j+(-window_size+i)]] for j in range(num_words) for i in range((2*window_size)+1) if ((-window_size+i)!=0 and j+(-window_size+i)>-1 and j+(-window_size+i)<num_words)]
    return (np.array(x_skipgram),np.array(y_skipgram))

if __name__ == '__main__':
  x_cbow, y_cbow = generate_training_data_cbow(words_in_order, word_to_index, 4)
  x_skipgram, y_skipgram = generate_training_data_skipgram(words_in_order, word_to_index, 4)
  print(x_cbow,y_cbow,x_skipgram,y_skipgram)

#One hot encoding
def one_hot(arr, num_vocabs):

  ra=(((arr.reshape(-1,1)==np.arange(num_vocabs))*1))
  return ra

def one_hot_multicategorical(arr, num_vocabs):
  shape=(arr.shape)
  ra=((arr.reshape(shape[0],-1,1)==(np.arange(num_vocabs)))*1)
  return (np.sum(ra,axis=1))

if __name__ == '__main__':
  x_cbow_1hot = one_hot_multicategorical(x_cbow, num_vocabs)
  y_cbow_1hot = one_hot(y_cbow, num_vocabs)
  x_skipgram_1hot = one_hot(x_skipgram, num_vocabs)
  y_skipgram_1hot = one_hot(y_skipgram, num_vocabs)

#Compiling the model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import Adam # Import Adam optimizer
from keras.metrics import categorical_crossentropy

def create_cbow_model():

  model = Sequential()
  model.add(Flatten(input_shape=(32,32)))
  model.add(Dense(units=4096,activation=None, use_bias=True))
  model.add(Dense(units=4096,activation=None, use_bias=True))
  model.add(Dense(units=4096,activation=None, use_bias=True))
  model.add(Dense(units=4096,activation=None, use_bias=True))

  model.add(Dense(units=10, activation='softmax', use_bias=True))
  adam_optimizer = Adam(learning_rate=0.001)

  model.compile(optimizer=adam_optimizer,loss=categorical_crossentropy, metrics=['accuracy'])
  return model


def create_skipgram_model(num_vocabs, dims_to_learn=50):
  model = Sequential()

  model.add(Dense(units=dims_to_learn,activation=None, use_bias=False, input_shape=(num_vocabs,)))
  model.add(Dense(units=num_vocabs, activation='softmax', use_bias=False))
  adam_optimizer = Adam(learning_rate=0.001)
  model.compile(optimizer=adam_optimizer,loss=categorical_crossentropy, metrics=['accuracy'])
  return model

if __name__ == "__main__":
  model_cbow = create_cbow_model()
  model_cbow.summary()
  model_skipgram = create_skipgram_model(num_vocabs)
  model_skipgram.summary()

#Training the model
if __name__ == '__main__':
  epochs = 500
  model1 = model_cbow.fit(x_cbow_1hot, y_cbow_1hot, epochs=epochs)
  model2 = model_skipgram.fit(x_skipgram_1hot, y_skipgram_1hot, epochs=epochs)

#Getting the output
def get_embeddings(words, weights, word_to_index, num_vocabs):

  a=[]
  for i in (words):
    index=word_to_index[i]
    myarray=(weights[index])
    a.append(myarray.tolist())
  return np.array(a)


  ### TO DO ###
  pass

#Visualizing the outcome
if __name__ == '__main__':
  import matplotlib.pyplot as plt  # for plotting graphs
  def plot_semantic_space(dim1, dim2, words, weights, word_to_index, num_vocabs):
    coordinates = get_embeddings(words, weights, word_to_index, num_vocabs)[:, [dim1,dim2]]
    x_points = coordinates[:, 0]
    y_points = coordinates[:, 1]


    plt.figure(figsize=(15, 15))
    plt.scatter(x_points, y_points)
    for i in range(0, len(words)):
      plt.annotate(text=words[i], xy=coordinates[i])

  # extract the weights from the model
  weights_cbow = model_cbow.layers[0].get_weights()[0]
  weights_skipgram = model_skipgram.layers[0].get_weights()[0]

  # visualization

  plot_semantic_space(10, 20, words_in_order, weights_skipgram,word_to_index, num_vocabs)
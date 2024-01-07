
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical


import matplotlib.pyplot as plt
import numpy as np
import os

def let_me_see(history):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True) # create a new folder "models" to save your model

# Loading the MNIST dataset and splitting it into training and testing sets
if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # x_train: (N, h, w) = (60000, 28, 28)
    # y_train: (N,) = (60000,)
    # x_test: (N, h, w) = (10000, 28, 28)
    # y_test: (N,) = (10000,)

#normalization
def process_x(x):
    x_normalized = (x/255)
    return x_normalized

#One-hot encoding
def process_y(y):
    y_onehot = keras.utils.to_categorical(y)

    return y_onehot

if __name__ == "__main__":
    # Performing data normalization

    x_train = process_x(x_train)
    x_test = process_x(x_test)

    # x_train: (N, h, w) = (60000, 28, 28)
    # x_test: (N, h, w) = (10000, 28, 28)

    # Converting the labels to one-hot encoded vectors
    y_train = process_y(y_train)
    y_test = process_y(y_test)

    # y_train: (N, 10) = (60000, 10)
    # y_test: (N, 10) = (10000, 10)

# visualize the picture in x_train
if __name__ == "__main__":
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(x_train[i], cmap=plt.cm.gray_r)
        plt.xticks([])
        plt.yticks([])
        plt.title(f"label:{np.argwhere(y_train[i] == 1)[0][0]}")

#Building a baseline model for comparison
def BaselineModel(img_shape):
    # Creating a sequential model
    model = Sequential()

    # Flattening the input image
    model.add(Flatten(input_shape=img_shape))

    # Adding the output layer with 10 units (one for each class) and softmax activation
    model.add(Dense(10, activation='softmax'))

    return model

#Buidling the actual model
def myModel(img_shape):
    model = Sequential()

    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=128, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))


    return model

#Training the baseline model
if __name__ == "__main__":
    baseline = BaselineModel(img_shape=(28, 28))

    baseline.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    baseline_history = baseline.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

#Evaluating the baseline model
if __name__ == "__main__":
    test_loss, test_accuracy = baseline.evaluate(x_test, y_test, verbose=1)
    print(f'Test accuracy: {test_accuracy}')

#Training my model
if __name__ == "__main__":
    model = myModel(img_shape=(28, 28))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join("models", "weights.hdf5"),
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True)

    history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, callbacks=[checkpointer])

#Evaluating my model
if __name__ == "__main__":
    # load the best model
    model = keras.models.load_model(os.path.join('models', 'weights.hdf5'))
    # Evaluate the model on the test data
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f'Test accuracy: {test_accuracy}')

#Visualizing the progress my model
if __name__ == "__main__":
    let_me_see(history)
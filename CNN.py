

from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
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

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Performing data normalization
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Converting the labels to one-hot encoded vectors
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    # x_train: (N, h, w, c) = (50000, 32, 32, 3)
    # y_train: (N, 10) = (50000, 10)
    # x_test: (N, h, w, c) = (10000, 32, 32, 3)
    # y_test: (N, 10) = (10000, 10)

if __name__ == "__main__":
    # There are 10 classes in CIFAR10 dataset
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

if __name__ == "__main__":
    # visualize the picture in x_train
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(x_train[i])
        plt.xticks([])
        plt.yticks([])
        plt.title(f"class{np.argwhere(y_train[i] == 1)[0][0]}: {classes[np.argwhere(y_train[i] == 1)[0][0]]}")

#some basic data augmentation
def Data_augmentation(img_shape):
    model = Sequential()
    model.add(RandomFlip(mode="horizontal"))

    return model

#Building the main CNN model after extensive trial and error
def myModel(img_shape=(32,32,3)):
    # Create a sequential model
    model = Sequential()

    # Add the data augmentation layer

    model.add(Conv2D(256, (3,3), padding='valid', activation='relu', input_shape=(32,32,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, (3,3), padding='valid', activation='relu', input_shape=(32,32,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3,3), padding='valid', activation='relu', input_shape=(32,32,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))

    model.add(Dense(10, activation='softmax'))

    print(model.summary())
    return model

model_x=myModel((32,32,3))
print(model_x.summary())

#Training the model
if __name__ == "__main__":
    model = myModel(img_shape=(32, 32, 3), data_aug=data_aug)
    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join("models", "weights.hdf5"),
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True)

    history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, callbacks=[checkpointer])

#Evaluating the model
if __name__ == "__main__":
    # load the best model
    model = keras.models.load_model(os.path.join('models', 'weights.hdf5'))
    # Evaluate the model on the test data
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f'Test accuracy: {test_accuracy}')

#Visualizing the history of the model
if __name__ == "__main__":
    let_me_see(history)
import pytest
import numpy as np
from cnn import CNN
from tensorflow.keras.datasets import fashion_mnist

def test_train_and_evaluate():	
    batch_size = 32
    num_classes = 10
    epochs = 100
    num_epochs = 30
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    number_of_train_samples_to_use = 100
    X_train = X_train[0:number_of_train_samples_to_use, :]
    y_train = y_train[0:number_of_train_samples_to_use]
    my_cnn=CNN()
    my_cnn.add_input_layer(shape=(28,28,1),name="input")
    my_cnn.append_conv2d_layer(num_of_filters=32, kernel_size=(3,3),padding="same", activation='linear')
    my_cnn.append_maxpooling2d_layer(pool_size=2, padding="same")
    my_cnn.append_conv2d_layer(num_of_filters=64, kernel_size=(3,3), activation='relu')
    my_cnn.append_maxpooling2d_layer(pool_size=2, padding="same")
    my_cnn.append_conv2d_layer(num_of_filters=128, kernel_size=(3,3), activation='relu')
    my_cnn.append_maxpooling2d_layer(pool_size=2, padding="same")
    my_cnn.append_flatten_layer(name="flatten")
    my_cnn.append_dense_layer(num_nodes=128,activation="relu",name="dense1")
    my_cnn.append_dense_layer(num_nodes= num_classes,activation="relu")
    out = my_cnn.predict(X_train)
    history = my_cnn.train(X_train, y_train, batch_size = batch_size, num_epochs = num_epochs)
    print("Accuracy of the model increases if we use Softmax instead of relu in the Dense layer!")
    test_eval = my_cnn.evaluate(X_test, y_test)
    print("Test loss = ", test_eval[0])
    print("Test accuracy = ", test_eval[1])

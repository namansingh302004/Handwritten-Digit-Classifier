import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from adjust_dataset import augment_image

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print('modifying data')

for i in range(x_train.shape[0]):
    x_train[i] = augment_image(x_train[i])

for i in range(x_test.shape[0]):
    x_test[i] = augment_image(x_test[i])

model.fit(x_train, y_train, epochs=15)

val_loss, val_acc = model.evaluate(x_test, y_test)
print("Validation loss:", val_loss)
print("Validation accuracy:", val_acc)

model.save('nn_model.h5')

new_model = keras.models.load_model('nn_model.h5')
predictions = new_model.predict(x_test)
print("First prediction: ", np.argmax(predictions[0]))

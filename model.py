from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Activation, Flatten, Dense
from keras import backend as k 
import numpy as np
import os
import matplotlib.pyplot as plt

train = ImageDataGenerator(rescale= 1/255)
validation = ImageDataGenerator(rescale= 1/255)

train_dataset = train.flow_from_directory('C:/Users/elyaa/Desktop/NEW_TIPE/data/train',
    target_size=(200,200),
    batch_size=20,
    class_mode='binary'
)
validation_dataset = train.flow_from_directory('C:/Users/elyaa/Desktop/NEW_TIPE/data/val',
    target_size=(200,200),
    batch_size=20,
    class_mode='binary'
)
print(train_dataset.class_indices)
#realisation du model machine learning
model = Sequential()
model.add(Conv2D(16, (3,3), input_shape = (200, 200, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(2, 2))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(2, 2))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(2, 2))

model.add(Flatten())

model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
           optimizer= 'rmsprop',
            metrics=['accuracy'])
m = model.fit(train_dataset,
        steps_per_epoch=8,
        epochs=500,
        validation_data=validation_dataset)
model.save('tipemls')


plt.plot(m.history['accuracy'])
plt.plot(m.history['val_accuracy'])
plt.title('Precision du modèle')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train-Entraînement', 'Validation'], loc='upper left')
plt.show()
plt.plot(m.history['loss'])
plt.plot(m.history['val_loss'])
plt.title('Perte du modèle')
plt.ylabel('loss-Perte')
plt.xlabel('epoch')
plt.legend(['Train-Entraînement', 'Validation'], loc='upper left')
plt.show()

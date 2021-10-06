import os
from keras.preprocessing import image
import matplotlib.pyplot as plt 
import numpy as np
from keras.utils.np_utils import to_categorical
import random,shutil
from keras.models import Sequential
from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, BatchNormalization
from keras.models import load_model


def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True,batch_size=1,target_size=(24,24),class_mode='categorical' ):
    return gen.flow_from_directory(dir,batch_size=batch_size,shuffle=shuffle,color_mode='grayscale',class_mode=class_mode,target_size=target_size)

BS= 32
TS=(24,24)
train_batch= generator('data/train',shuffle=True, batch_size=BS,target_size=TS)
valid_batch= generator('data/valid',shuffle=True, batch_size=BS,target_size=TS)
SPE= len(train_batch.classes)//BS
VS = len(valid_batch.classes)//BS
print(SPE,VS)


model = Sequential([

    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,1)), #Capa convolusional 1
    MaxPooling2D(pool_size=(1,1)),


    Conv2D(filters=32,kernel_size=(3,3),activation='relu'), #Capa convolusional 2
    MaxPooling2D(pool_size=(1,1)),


    Conv2D(filters=64,kernel_size=(3, 3),activation='relu'), #Capa convolusional 3
    MaxPooling2D(pool_size=(1,1)),

    

    Dropout(0.25), #Apagamos el 25% de las imagenes a cada paso

    Flatten(), #Aplanamos la imagen

    Dense(128, activation='relu'), 
    #Añadimos una capa que conectara la anterior y la siguiente con 128 conexiones

    Dropout(0.5), #Apagamos el 50% de las imagenes a cada paso

    Dense(4, activation='softmax') #Capa softmax

])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit_generator(train_batch, validation_data=valid_batch,epochs=15,steps_per_epoch=SPE ,validation_steps=VS)

model.save('models/cnnCat2.h5', overwrite=True)
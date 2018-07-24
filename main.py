import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import shutil
import pickle

import random

TRAIN_PATH='./train'
TEST_PATH='./test1'

train_cats=os.path.join(TRAIN_PATH,'cats')
#os.makedirs(train_cats)
train_dogs=os.path.join(TRAIN_PATH,'dogs')
#os.makedirs(train_dogs)

len_cats=len(os.listdir(train_cats))
len_dogs=len(os.listdir(train_dogs))

SIZE=len_dogs+len_cats
val_size=int(0.2*SIZE/2)

files_cats=os.listdir(train_cats)
files_dogs=os.listdir(train_dogs)

validation_path='./validation'
#os.makedirs(validation_path)
validation_cats=os.path.join(validation_path,'cats')
#os.makedirs(validation_cats)
validation_dogs=os.path.join(validation_path,'dogs')
#os.makedirs(validation_dogs)
val_len_dogs=len(os.listdir(validation_dogs))
val_len_cats=len(os.listdir(validation_cats))
VAL_SIZE=val_len_dogs+val_len_cats

datagen=ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=4,height_shift_range=4,horizontal_flip=True,vertical_flip=False)

train_generator=datagen.flow_from_directory(TRAIN_PATH,target_size=[150,150],batch_size=16,class_mode='binary',interpolation='nearest')

datagen_val=ImageDataGenerator(rescale=1./255)
validation_generator=datagen_val.flow_from_directory(validation_path,target_size=[150,150],batch_size=16,class_mode='binary',interpolation='nearest')


model=tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(16,3,activation='relu',input_shape=[150,150,3],name='conv1'))
model.add(tf.keras.layers.MaxPooling2D((2,2),name='pool1'))
model.add(tf.keras.layers.BatchNormalization(axis=-1,name='batch_norm1'))
model.add(tf.keras.layers.Conv2D(32,3,activation='relu',name='conv2'))
model.add(tf.keras.layers.MaxPooling2D((2,2),name='pool2'))
model.add(tf.keras.layers.BatchNormalization(name='batch_norm2'))
model.add(tf.keras.layers.Conv2D(32,3,activation='relu',name='conv3'))
model.add(tf.keras.layers.MaxPooling2D((2,2),name='pool3'))
model.add(tf.keras.layers.Conv2D(64,3,activation='relu',name='conv4'))
model.add(tf.keras.layers.BatchNormalization(name='batch_norm4'))
model.add(tf.keras.layers.MaxPooling2D((2,2),name='pool4'))
model.add(tf.keras.layers.Flatten(name='flatten'))
model.add(tf.keras.layers.Dense(100,activation='relu',name='dense1'))
model.add(tf.keras.layers.Dense(1,activation='sigmoid',name='output'))
model.summary()

num_steps_per_epoch=SIZE/16 
val_num_steps_per_epoch=VAL_SIZE/16

callbacks_list=[ModelCheckpoint(filepath='./neww/model.{epoch:02d}-{val_acc:.2f}.hdf5')]


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history =  model.fit_generator(train_generator,steps_per_epoch=num_steps_per_epoch,epochs=10,validation_data=validation_generator,validation_steps=val_num_steps_per_epoch,callbacks=callbacks_list,verbose=1)

tf.keras.models.save_model(model,'./neww/model.h5')



f=open('history','wb')
pickle.dump(f,history.history)


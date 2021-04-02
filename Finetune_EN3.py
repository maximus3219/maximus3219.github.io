import os, shutil
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


new_model = tf.keras.models.load_model('G:/Myxoid lesion/Myxoid_EN3')


#All layers become training except for the BatchNormalization layers
new_model.summary()

data_dir='G:/Myxoid lesion/Training/'

batch_size = 64
img_height = 300
img_width = 300

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=16479,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=16479,
  image_size=(img_height, img_width),
  batch_size=batch_size)

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=10)

rlr = ReduceLROnPlateau(monitor='val_loss', 
                        factor=0.5, 
                        patience=4, 
                        verbose=1, 
                        mode='auto', 
                        min_delta=0.0001)

mc = ModelCheckpoint("G:/Myxoid lesion/Myxoid_EN3_finetune/",
                     monitor='val_loss', mode='auto', verbose=1,
                     save_best_only=True)

new_model.compile(optimizer = optimizers.Adam(lr=0.00002),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics = ['sparse_categorical_accuracy'])

epochs=10
history_fine = new_model.fit(train_ds,
                        epochs=epochs,
                         validation_data=val_ds,
                          verbose=2,
                         callbacks=[es, rlr, mc])

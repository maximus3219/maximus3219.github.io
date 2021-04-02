import tensorflow as tf
import os, shutil
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import efficientnet.tfkeras as efn


batch_size = 64
img_height = 300
img_width = 300

data_dir= 'G:/Myxoid lesion/Training'

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)


num_classes = 5

#Data augmentation
data_augmentation = tf.keras.Sequential(
  [layers.experimental.preprocessing.RandomFlip(mode="horizontal_and_vertical", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.2),
    layers.experimental.preprocessing.RandomZoom(-0.2, 0.2),
    layers.experimental.preprocessing.RandomTranslation(
    height_factor=(-0.2, 0.2), width_factor =(-0.2, 0.2)),
   layers.experimental.preprocessing.RandomContrast(0.1)
  ])

# Crease base model
base_model = efn.EfficientNetB3(input_shape = (img_height, img_width, 3),
                                include_top = False,
                                weights = 'G:/Deeplearning/Base model/Efficientnet/efficientnet-b3_noisy-student_notop.h5')

# Freeze base model
for layer in base_model.layers:
    layer.trainable = False


# Create model
model = models.Sequential([
    data_augmentation,
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(256, activation='relu'),
    layers.Dense(num_classes)
    ])


base_learning_rate = 0.0001

model.compile(optimizer= optimizers.Adam(lr=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])


model.summary()


es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=12)
rlr = ReduceLROnPlateau(monitor='val_loss', 
                        factor=0.5, 
                        patience=4, 
                        verbose=1, 
                        mode='auto', 
                        min_delta=0.0001)

mc = ModelCheckpoint("G:/Myxoid lesion/Myxoid_EN3/",
                     monitor='val_loss', mode='auto', verbose=1,
                     save_best_only=True)

epochs=20
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  verbose=2,
  callbacks=[es, rlr, mc]
)

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(epochs_range,val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

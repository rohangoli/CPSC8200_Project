## Purpose: ResNetv50 Training in serial execution
## Import required Modules
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import argparse
import time
import sys

from cifar import load_cifar

## Add arguments to the python script
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=256)

args = parser.parse_args()
batch_size = args.batch_size
epochs = args.epochs

## Load dataset
train_ds, test_ds = load_cifar(batch_size)

## Initiate a ResNetv50 Model
model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=True, weights=None,
            input_shape=(128, 128, 3), classes=10)

## Initiate Optimizer
opt = tf.keras.optimizers.SGD(0.01)

## Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

## Train the model
start = time.time()
model.fit(train_ds, epochs=epochs, verbose=1)
end = time.time()

print("Total Time Taken: ", end-start)
print('Avg per epoch:', round((end - start)/epochs, 2))

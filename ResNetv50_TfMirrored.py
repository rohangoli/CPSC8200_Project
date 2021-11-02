## Include required modules
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import argparse
import time
import sys

from cifar import load_cifar

## Enable arguments over python script
parser = argparse.ArgumentParser()
## No. of time data should be trained on the model
parser.add_argument('--epochs', type=int, default=5)
## Mention Batch Size as 256 * No.of GPUS
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--n_gpus', type=int, default=1)

## Parse arguments
args = parser.parse_args()
batch_size = args.batch_size
epochs = args.epochs
n_gpus = args.n_gpus

## Load Dataset
train_ds, test_ds = load_cifar(batch_size)

## List GPU Devices
device_type = 'GPU'
devices = tf.config.experimental.list_physical_devices(device_type)
devices_names = [d.name.split("e:")[1] for d in devices]

## Declare Strategy
strategy = tf.distribute.MirroredStrategy(devices=devices_names[:n_gpus])

## Declare and Compile model in the scope of strategy
with strategy.scope():
    model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=True, weights=None,
            input_shape=(128, 128, 3), classes=10)
    ## Increase learning with increase in gpus to converge faster
    opt = tf.keras.optimizers.SGD(0.01*n_gpus)

    model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

## Train the model
start = time.time()
model.fit(train_ds, epochs=epochs, verbose=1)
end = time.time()

print("Total Time Taken: ", end-start)
print('Avg per epoch:', round((end - start)/epochs, 2))

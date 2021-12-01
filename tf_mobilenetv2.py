import tensorflow as tf
import tensorflow_datasets as tfds
import time, json, os

tf_config = {
    'cluster': {
       'worker': ['node1782.palmetto.clemson.edu']
                    
    },
    'task': {'type': 'worker', 'index': 0}
}
json.dumps(tf_config)
per_worker_batch_size = 16
#tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])
global_batch_size = per_worker_batch_size * num_workers
print(global_batch_size)
#multi_worker_dataset = mnist.mnist_dataset(global_batch_size)

communication_options = tf.distribute.experimental.CommunicationOptions(
    implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)
#strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=communication_options)

num_epochs = 10
#batch_size_per_replica = 16
learning_rate = 0.001

strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=communication_options)
print('Number of devices: %d' % strategy.num_replicas_in_sync) 

def resize(image, label):
    image = tf.image.resize(image, [224, 224]) / 255.0
    return image, label

dataset = tfds.load("cats_vs_dogs", split=tfds.Split.TRAIN, as_supervised=True)
dataset = dataset.map(resize).shuffle(1024).batch(global_batch_size)

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

with strategy.scope():
    model = tf.keras.applications.MobileNetV2(weights=None, classes=2)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )

startTime = time.time()

history=model.fit(dataset, epochs=num_epochs, callbacks=[callback])

endTime = time.time()

print("Exec Time: ", endTime-startTime)
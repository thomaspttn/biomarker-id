from kerastuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

df = pd.read_csv('./dl-classifier/brca-normalized.csv')
print(df)

# x = all cols but last, y = last col
x = df.iloc[:, :-1]
y = df.iloc[:, -1].astype('float32')

# print(df['24'].value_counts())


"""
def build_model(hp):
    model = tf.keras.Sequential()
    for i in range(hp.Int('num_layers', 2, 20)):
        model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=1,
                                            max_value=512,
                                            step=16),
                                            activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=50,
    executions_per_trial=7,
    directory='my_dir'
)

tuner.search(
    x,
    y,
    epochs=500,
    validation_split=0.2
)

models = tuner.get_best_models(num_models=2)
print(models)
print(tuner.results_summary())
"""

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu', dtype='float32'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu', dtype='float32'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu', dtype='float32'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu', dtype='float32'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')
])

adam = tf.keras.optimizers.Adam(learning_rate=0.00001)
model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

model.fit(x,
          y,
          batch_size=1,
          epochs=500,
          validation_split=0.2,
          verbose=1)

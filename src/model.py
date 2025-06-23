import tensorflow as tf
from tensorflow.keras import layers, models

# Proposed Model
class Critic(tf.keras.Model):
    def __init__(self, input_shape):
        super(Critic, self).__init__()
        self.lstm = layers.LSTM(64, return_sequences=False)
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(1, activation='linear')

    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.dense1(x)
        return self.dense2(x)

class Actor(tf.keras.Model):
    def __init__(self, input_shape, num_features, steps_ahead):
        super(Actor, self).__init__()
        self.lstm1 = layers.LSTM(312, return_sequences=True)
        self.bilstm = layers.Bidirectional(layers.LSTM(312, return_sequences=True))
        self.lstm2 = layers.LSTM(128, return_sequences=True)
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(num_features, activation='linear')
        self.reshape = layers.Reshape((steps_ahead, num_features))

    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.bilstm(x)
        x = self.lstm2(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.reshape(x)

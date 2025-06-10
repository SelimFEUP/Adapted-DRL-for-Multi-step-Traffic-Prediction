import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.utils import plot_model

import random

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# Load the dataset
def load_data(filepath):
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
    # Handle missing values if necessary (e.g., fill forward/backward)
    data.fillna(method='ffill', inplace=True)
    data.interpolate(method='linear', inplace=True)
    return data.values

# Preprocess the data: Scaling, converting to numpy array
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

# Create windows for input-output pairs (for multi-step time series forecasting)
def create_sequences(data, input_steps, output_steps):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps):
        X.append(data[i:(i + input_steps), :])
        y.append(data[(i + input_steps):(i + input_steps + output_steps), :])
    return np.array(X), np.array(y)

# Split the data into training, validation, and testing sets
def split_data(X, y, test_size=0.2, val_size=0.1):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size/(test_size + val_size))
    return X_train, X_val, X_test, y_train, y_val, y_test
    
    
# Novel DRL Model
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
        
# Train the DRL-based multi-step prediction model
def train_drl_model(X_train, Y_train, X_val, Y_val, steps_ahead, num_features):
    input_shape = X_train.shape[1:]  # (sequence_length, num_features)
    output_shape = Y_train.shape[1]  # steps_ahead

    actor = Actor(input_shape, num_features, steps_ahead)
    critic = Critic(input_shape)

    actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    mse_loss = tf.keras.losses.MeanSquaredError()

    best_val_mae = float('inf')  # Initialize the best validation MAE

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            actions = actor(x)
            q_values = critic(x)
            predicted_q = -mse_loss(y, actions) + tf.reduce_mean(q_values)

            actor_loss = -predicted_q  # Maximize Q
            critic_loss = mse_loss(y, q_values)

        actor_gradients = actor_tape.gradient(actor_loss, actor.trainable_variables)
        critic_gradients = critic_tape.gradient(critic_loss, critic.trainable_variables)

        actor_optimizer.apply_gradients(zip(actor_gradients, actor.trainable_variables))
        critic_optimizer.apply_gradients(zip(critic_gradients, critic.trainable_variables))

    # Training loop
    epochs = 120
    for epoch in range(epochs):
        for i in range(len(X_train)):
            train_step(X_train[i:i+1], Y_train[i:i+1])

        # Validation
        val_predictions = actor(X_val).numpy()
        val_predictions = np.reshape(val_predictions, Y_val.shape)

        # Compute metrics
        mae = mean_absolute_error(Y_val.flatten(), val_predictions.flatten())
        rmse = np.sqrt(mean_squared_error(Y_val.flatten(), val_predictions.flatten()))

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Validation MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        # Save the actor and critic models if the validation MAE improves
        if mae < best_val_mae:
            best_val_mae = mae
            actor.save_weights('best_actor_model_pems_24.weights.h5')
            critic.save_weights('best_critic_model_pems_24.weights.h5')
            print(f"Best models saved with Validation MAE: {best_val_mae:.4f}")

    return actor, critic


def validate_model(actor_model, X_val, Y_val):
    # Predict and calculate metrics
    predictions = actor_model.predict(X_val)
    
    # Flatten the predicted and actual values to compute MAE and RMSE
    Y_val_flat = Y_val.reshape(-1, num_features)
    predictions_flat = predictions.reshape(-1, num_features)
    
    mae = mean_absolute_error(Y_val_flat, predictions_flat)
    rmse = np.sqrt(mean_squared_error(Y_val_flat, predictions_flat))
    
    print(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}%")
    
    return mae, rmse


# Main pipeline
filepath = 'data/transformed_data.csv'  # Update to your dataset path

data = load_data(filepath)

input_steps = 24
output_steps = 24
# Create sequences for multi-step forecasting
X, y = create_sequences(data, input_steps=input_steps, output_steps=output_steps)
    
# Split the data into training, validation, and test sets
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
scaler = MinMaxScaler(feature_range=(0, 1))
    
# Reshape to 2D for scaling
num_features = X_train.shape[2]
X_train_reshaped = X_train.reshape(-1, num_features)
X_val_reshaped = X_val.reshape(-1, num_features)
X_test_reshaped = X_test.reshape(-1, num_features)
y_train_reshaped = y_train.reshape(-1, num_features)
y_val_reshaped = y_val.reshape(-1, num_features)
y_test_reshaped = y_test.reshape(-1, num_features)
    
# Fit scaler on training data
scaler.fit(X_train_reshaped)
    
# Transform all data
X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
X_val_scaled = scaler.transform(X_val_reshaped).reshape(X_val.shape)
X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)
y_train_scaled = scaler.transform(y_train_reshaped).reshape(y_train.shape)
y_val_scaled = scaler.transform(y_val_reshaped).reshape(y_val.shape)
y_test_scaled = scaler.transform(y_test_reshaped).reshape(y_test.shape)

actor_model, critic_model = train_drl_model(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, steps_ahead=output_steps, num_features=num_features)

# Load the weights into the models
actor_model.load_weights('models/best_actor_model_pems_24.weights.h5')
critic_model.load_weights('models/best_critic_model_pems_24.weights.h5')


validate_model(actor_model, X_test_scaled, y_test_scaled)

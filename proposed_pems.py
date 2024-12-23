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
       
    # Handle missing values
    data.fillna(method='ffill', inplace=True)
    data.interpolate(method='linear', inplace=True)
    return data.values

# Preprocessing
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

# Creating windows
def create_sequences(data, input_steps, output_steps):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps):
        X.append(data[i:(i + input_steps), :])
        y.append(data[(i + input_steps):(i + input_steps + output_steps), :])
    return np.array(X), np.array(y)

# Spliting the data into training, validation, and testing sets
def split_data(X, y, test_size=0.2, val_size=0.1):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size/(test_size + val_size))
    return X_train, X_val, X_test, y_train, y_val, y_test
    
    
# DRL Model
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

class Actor_reshape(tf.keras.Model):
    def __init__(self, input_shape, num_features, steps_ahead):
        super(Actor, self).__init__()
        self.lstm1 = layers.LSTM(312, return_sequences=True)
        self.bilstm = layers.Bidirectional(layers.LSTM(312, return_sequences=True))
        self.lstm2 = layers.LSTM(128, return_sequences=False)  # Change to return_sequences=False
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(steps_ahead * num_features, activation='linear') 
        self.reshape = layers.Reshape((steps_ahead, num_features))

    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.bilstm(x)
        x = self.lstm2(x) 
        x = self.dense1(x)
        x = self.dense2(x) 
        return self.reshape(x)
        
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

def ces(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred))

# Training
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
    epochs = 500
    for epoch in range(epochs):
        for i in range(len(X_train)):
            train_step(X_train[i:i+1], Y_train[i:i+1])

        # Validation
        val_predictions = actor(X_val).numpy()
        val_predictions = np.reshape(val_predictions, Y_val.shape)

        # Compute metrics
        mae = mean_absolute_error(Y_val.flatten(), val_predictions.flatten())
        rmse = np.sqrt(mean_squared_error(Y_val.flatten(), val_predictions.flatten()))
        ces_score = ces(Y_val.flatten(), val_predictions.flatten())

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Validation MAE: {mae:.4f}, RMSE: {rmse:.4f}, CES_score: {ces_score:.4f}")

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
       
    print(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    return mae, rmse


filepath = 'transformed_data.csv'  # Update to your dataset path

data = load_data(filepath)

input_steps = 24
output_steps = 24

# Create sequences for multi-step-ahead prediction
X, y = create_sequences(data, input_steps=input_steps, output_steps=output_steps)
    
# Spliting the data
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

# Rebuild the Actor and Critic models with the same architecture
#actor_model = Actor(input_shape=X_train_scaled.shape[1:], num_features=num_features, steps_ahead=output_steps)
#critic_model = Critic(input_shape=X_train_scaled.shape[1:])

# Load the weights into the models
actor_model.load_weights('best_actor_model_pems_24.weights.h5')
critic_model.load_weights('best_critic_model_pems_24.weights.h5')


validate_model(actor_model, X_test_scaled, y_test_scaled)

#### Plotting the results

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Disable external LaTeX, but keep math text rendering
rcParams['text.usetex'] = False  # Use mathtext instead of full LaTeX
rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern, the default LaTeX font

# Reshape to 2D for inverse transform
Y_test_reshaped = y_test_scaled.reshape(-1, num_features)
predicted_values_scaled = actor_model.predict(X_test_scaled).reshape(-1, num_features)

# Inverse transform
true_values = scaler.inverse_transform(Y_test_reshaped)  # Original scale
predicted_values = scaler.inverse_transform(predicted_values_scaled)

# Reshape back to original shape if needed
true_values = true_values.reshape(y_test_scaled.shape)
predicted_values = predicted_values.reshape(y_test_scaled.shape)

# Sensors to plot (update indices based on your dataset's columns)
sensor_indices = [13, 5, 7, 11]  # Randomly selected

# Plot for each sensor
annotations = ['(a)', '(b)', '(c)', '(d)'] 
plt.figure(figsize=(7, 7))

for i, sensor_idx in enumerate(sensor_indices):
    plt.subplot(2, 2, i + 1)  # Create a 2x2 grid for plots
    plt.scatter(
        true_values[:, 0, sensor_idx],
        predicted_values[:, 0, sensor_idx],
        color='blue',
        alpha=0.7,
        s=5  # Adjust the marker size (smaller dots)
    )
    plt.plot(
        [np.min(true_values[:, 0, sensor_idx]), np.max(true_values[:, 0, sensor_idx])],
        [np.min(true_values[:, 0, sensor_idx]), np.max(true_values[:, 0, sensor_idx])],
        color='red', linestyle='dashed'
    )  # 45° line
    #plt.title(f'Sensor {sensor_idx}')
    plt.ylabel('Predicted Values')

    # Set y-axis label with annotation in the middle
    plt.xlabel(f'Actual Values\n{annotations[i]}', fontsize=10)

plt.tight_layout()  # Adjust layout for better viewing
plt.show()

for i, sensor_idx in enumerate(sensor_indices):
    plt.subplot(2, 2, i + 1)  # Create a 2x2 grid for plots
    # Plot True values as a line
    plt.plot(true_values[:48, 0, sensor_idx], label='True', color='b', linestyle='-')  # True values in blue

    # Plot Predicted values as a transparent line
    plt.plot(predicted_values[:48, 0, sensor_idx], label='Predicted', color='r', linestyle='-', alpha=0.3) 
    # Plot Predicted values with markers ('*')
    plt.plot(predicted_values[:48, 0, sensor_idx], '*', color='r')  # Predicted values as red stars

    plt.ylabel('Traffic Flow')

    # Set x-axis label with annotation in the middle
    plt.xlabel(f'Time Steps\n{annotations[i]}', fontsize=10)

    # Add legend for each subplot
    plt.legend()

plt.tight_layout()  # Adjust layout for better viewing
plt.show()

# Defining evaluation functions (MAE, RMSE)
def evaluate_model(model, x_test, y_test):
    """
    Evaluate the model on the given test set using MAE and RMSE.
    """
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test.flatten(), y_pred.flatten())
    rmse = np.sqrt(mean_squared_error(y_test.flatten(), y_pred.flatten()))
    return mae, rmse
    
# Function to add noise to the dataset
def add_noise(data, noise_level):
    noise = np.random.normal(0, noise_level, data.shape)  # Gaussian noise
    noisy_data = data + noise
    return noisy_data

# Evaluate the model on the test set with various noise levels
def evaluate_with_noise(model, X_test, y_test, noise_levels):
    results = {}
    for noise_level in noise_levels:
        # Add noise to both input and output data
        noisy_X_test = add_noise(X_test, noise_level)
        noisy_y_test = add_noise(y_test, noise_level)
        
        # Evaluate the model
        mae, rmse = evaluate_model(model, noisy_X_test, noisy_y_test)
        
        # Store results
        results[noise_level] = {'MAE': mae, 'RMSE': rmse}
        print(f"Noise Level: {noise_level:.2f} => Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}")
    
    return results

# Noise levels
noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

results = evaluate_with_noise(actor_model, X_test_scaled, y_test_scaled, noise_levels)

# Function to simulate sudden increase and decrease in traffic flow
def simulate_traffic_changes(data, increase_idx, decrease_idx, increase_factor=3.5, decrease_factor=0.3):
    data_copy = data.copy()
    
    # Simulate sudden increase (at 'increase_idx' index)
    data_copy[increase_idx] = data_copy[increase_idx] * increase_factor
    
    # Simulate sudden decrease (at 'decrease_idx' index)
    data_copy[decrease_idx] = data_copy[decrease_idx] * decrease_factor
    
    return data_copy

# Indices for increase and decrease
increase_idx = [55, 100, 155]  # Choose an index for the increase
decrease_idx = [75, 185, 200]  # Choose an index for the decrease

# Simulate the traffic data with sudden changes
noisy_traffic_data = simulate_traffic_changes(X_test_scaled, increase_idx, decrease_idx)

# Evaluate the model before and after the changes
mae_before, rmse_before = evaluate_model(actor_model, X_test_scaled, y_test_scaled)
mae_after, rmse_after = evaluate_model(actor_model, noisy_traffic_data, y_test_scaled)

print(f"Before Change => Test MAE: {mae_before:.4f}, Test RMSE: {rmse_before:.4f}")
print(f"After Change => Test MAE: {mae_after:.4f}, Test RMSE: {rmse_after:.4f}")

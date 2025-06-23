import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from src.model import Actor, Critic
import random

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

       
# Train the proposed model
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
            actor.save_weights('./models/best_actor_model_pems_24.weights.h5')
            critic.save_weights('./models/best_critic_model_pems_24.weights.h5')
            print(f"Best models saved with Validation MAE: {best_val_mae:.4f}")

    return actor, critic

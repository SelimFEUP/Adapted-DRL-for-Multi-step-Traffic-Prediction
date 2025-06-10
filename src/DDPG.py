
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm


# GPU Configuration
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Environment Class
class MultiStepTrafficEnv:
    def __init__(self, data, n_steps, pred_steps):
        self.data = data
        self.n_steps = n_steps
        self.pred_steps = pred_steps
        self.n_sensors = data.shape[1]
        self.reset()

    def reset(self):
        self.current_step = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        return self.data[self.current_step:self.current_step + self.n_steps]

    def step(self, action):
        if self.done:
            raise ValueError("Environment is done. Call reset().")

        self.current_step += 1
        if self.current_step + self.n_steps + self.pred_steps >= len(self.data):
            self.done = True

        true_values = self.data[self.current_step + self.n_steps:
                                self.current_step + self.n_steps + self.pred_steps]
        reward = -mean_absolute_error(true_values.flatten(), action.flatten())

        next_state = self._get_state()
        return next_state, reward, self.done

# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size

    def add(self, experience):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

# DDPG Agent
class DDPG:
    def __init__(self, state_dim, action_dim, action_bound, lr_actor=0.001, lr_critic=0.002):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.target_actor = self.build_actor()
        self.target_critic = self.build_critic()

        self.actor_optimizer = tf.keras.optimizers.Adam(lr_actor)
        self.critic_optimizer = tf.keras.optimizers.Adam(lr_critic)

        self.update_target_weights(tau=1.0)

    def build_actor(self):
        inputs = tf.keras.layers.Input(shape=(self.state_dim,))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.action_dim, activation='sigmoid')(x)
        return tf.keras.Model(inputs, outputs)

    def build_critic(self):
        state_input = tf.keras.layers.Input(shape=(self.state_dim,))
        action_input = tf.keras.layers.Input(shape=(self.action_dim,))
        x = tf.keras.layers.Concatenate()([state_input, action_input])
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1)(x)
        return tf.keras.Model([state_input, action_input], outputs)
        
    def save(self, actor_path, critic_path, target_actor_path, target_critic_path):
        self.actor.save(actor_path)
        self.critic.save(critic_path)
        self.target_actor.save(target_actor_path)
        self.target_critic.save(target_critic_path)
        
    def load(self, actor_path, critic_path, target_actor_path, target_critic_path):
        self.actor = tf.keras.models.load_model(actor_path)
        self.critic = tf.keras.models.load_model(critic_path)
        self.target_actor = tf.keras.models.load_model(target_actor_path)
        self.target_critic = tf.keras.models.load_model(target_critic_path)

    def update_target_weights(self, tau=0.005):
        for target_param, param in zip(self.target_actor.variables, self.actor.variables):
            target_param.assign(tau * param + (1 - tau) * target_param)

        for target_param, param in zip(self.target_critic.variables, self.critic.variables):
            target_param.assign(tau * param + (1 - tau) * target_param)

    def train(self, replay_buffer, batch_size, gamma=0.99):
        batch = replay_buffer.sample(batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states = tf.convert_to_tensor(np.array(states).reshape(batch_size, -1), dtype=tf.float32)
        actions = tf.convert_to_tensor(np.array(actions).reshape(batch_size, -1), dtype=tf.float32)
        rewards = tf.convert_to_tensor(np.array(rewards).reshape(batch_size, 1), dtype=tf.float32)
        next_states = tf.convert_to_tensor(np.array(next_states).reshape(batch_size, -1), dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_states)
            target_q_values = rewards + gamma * self.target_critic([next_states, target_actions])
            target_q_values = tf.stop_gradient(target_q_values)
            q_values = self.critic([states, actions])
            critic_loss = tf.reduce_mean(tf.square(target_q_values - q_values))

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic([states, actions]))

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        self.update_target_weights()


# Training Setup
data = pd.read_csv("data/transformed_data.csv")
data.fillna(method='ffill', inplace=True)
data.interpolate(method='linear', inplace=True)
data = data.drop(columns=["date_hour"]).values

scaler = MinMaxScaler()
data = scaler.fit_transform(data)

n_steps = 24
pred_steps = 24
batch_size = 32
episodes = 500

state_dim = n_steps * data.shape[1]
action_dim = pred_steps * data.shape[1]
action_bound = 1.0

replay_buffer = ReplayBuffer(max_size=5000)
env = MultiStepTrafficEnv(data, n_steps, pred_steps)
agent = DDPG(state_dim, action_dim, action_bound)

# Training Loop
for episode in tqdm(range(episodes)):
    state = env.reset()
    state_flat = state.flatten()
    episode_reward = 0

    while not env.done:
        action = agent.actor.predict(np.expand_dims(state_flat, axis=0), verbose=0)[0]
        action = action * action_bound

        next_state, reward, done = env.step(action)
        next_state_flat = next_state.flatten()

        replay_buffer.add((state_flat, action, reward, next_state_flat))

        if len(replay_buffer.buffer) >= batch_size:
            agent.train(replay_buffer, batch_size)

        state_flat = next_state_flat
        episode_reward += reward

    print(f"Episode {episode + 1}/{episodes}, Reward: {episode_reward:.2f}")


agent.save(actor_path="models/actor_model.h5",critic_path="models/critic_model.h5",target_actor_path="models/target_actor_model.h5",
           target_critic_path="models/target_critic_model.h5")

# Evaluation
def evaluate_model(env, agent, scaler):
    state = env.reset()
    predictions, true_values = [], []

    while not env.done:
        state_flat = state.flatten()
        action = agent.actor.predict(np.expand_dims(state_flat, axis=0), verbose=0)[0]
        action = scaler.inverse_transform(action.reshape(pred_steps, -1))
        predictions.append(action)

        true_values.append(
            scaler.inverse_transform(
                env.data[env.current_step + n_steps:env.current_step + n_steps + pred_steps]
            )
        )

        state, _, _ = env.step(action)

    predictions = np.vstack(predictions)
    true_values = np.vstack(true_values)

    mae = mean_absolute_error(true_values, predictions)
    rmse = np.sqrt(mean_squared_error(true_values, predictions))
    return mae, rmse

mae, rmse = evaluate_model(env, agent, scaler)
print(f"Evaluation - MAE: {mae:.4f}, RMSE: {rmse:.4f}")



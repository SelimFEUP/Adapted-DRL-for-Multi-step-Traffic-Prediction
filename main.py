from src.preprocessing import load_data, output_steps, num_features
from src.train import train_drl_model
from src.evaluate import validate_model

X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled = load_data()

actor_model, critic_model = train_drl_model(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, steps_ahead=output_steps, num_features=num_features)

# Load the weights into the models
actor_model.load_weights('./models/best_actor_model_pems_24.weights.h5')

validate_model(actor_model, X_test_scaled, y_test_scaled)

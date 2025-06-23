import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.preprocessing import num_features

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

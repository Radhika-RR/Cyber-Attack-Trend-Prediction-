import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import joblib
import config

def evaluate_model():
    # Load data
    df = pd.read_csv(config.DATA_PATH)
    df['time_index'] = range(len(df))
    X = df[['time_index']]
    y = df['Type']
    
    # Load model
    model = joblib.load(config.MODEL_PATH)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    print("=== COMPREHENSIVE EVALUATION ===")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Calculate attack rate
    actual_attack_rate = y.mean()
    predicted_attack_rate = y_pred.mean()
    
    print(f"\nActual Attack Rate: {actual_attack_rate:.4f}")
    print(f"Predicted Attack Rate: {predicted_attack_rate:.4f}")
    
    return r2, mae, rmse

if __name__ == "__main__":
    evaluate_model()
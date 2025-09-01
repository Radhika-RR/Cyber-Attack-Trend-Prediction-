import joblib
import numpy as np
import config

def predict_future_attacks():
    # Load trained model
    model = joblib.load(config.MODEL_PATH)
    
    # Get the latest time index from training data
    latest_time = 1998  # Based on your dataset size
    
    # Predict next 10 time periods
    future_periods = 10
    future_times = np.array(range(latest_time + 1, latest_time + future_periods + 1))
    future_times = future_times.reshape(-1, 1)
    
    predictions = model.predict(future_times)
    
    print("=== FUTURE ATTACK PREDICTIONS ===")
    for i, pred in enumerate(predictions):
        print(f"Time period {latest_time + i + 1}: {pred:.4f} (expected attacks)")
    
    return predictions

if __name__ == "__main__":
    predict_future_attacks()
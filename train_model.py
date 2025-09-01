import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import config

# Load and prepare data
def load_data():
    df = pd.read_csv(config.DATA_PATH)
    
    # Create time-based features (assuming rows are chronological)
    df['time_index'] = range(len(df))
    
    # For time series prediction, we'll use time index as feature
    # and number of attacks (Type=1) as target
    X = df[['time_index']]
    y = df['Type']  # Binary classification (0/1) but we'll treat as count
    
    return X, y, df

def train_linear_regression():
    X, y, df = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("=== MODEL COEFFICIENTS ===")
    print(f"Coefficient: {model.coef_[0]}")
    print(f"Intercept: {model.intercept_}")
    print("\n=== EVALUATION METRICS ===")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Save model
    joblib.dump(model, config.MODEL_PATH)
    
    return model, X, y, df, X_test, y_test, y_pred

def plot_results(model, X, y, X_test, y_test, y_pred):
    plt.figure(figsize=(12, 6))
    
    # Plot actual data
    plt.scatter(X, y, alpha=0.5, label='Actual Data', color='blue')
    
    # Plot regression line
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_range = model.predict(x_range)
    plt.plot(x_range, y_range, color='red', linewidth=2, label='Regression Line')
    
    # Plot predictions
    plt.scatter(X_test, y_pred, alpha=0.7, label='Predictions', color='green')
    
    plt.xlabel('Time Index')
    plt.ylabel('Attack Type (0=Normal, 1=Malicious)')
    plt.title('Cyber Attack Trend Prediction using Linear Regression')
    plt.legend()
    plt.grid(True)
    plt.savefig('attack_trend_prediction.png')
    plt.show()

if __name__ == "__main__":
    model, X, y, df, X_test, y_test, y_pred = train_linear_regression()
    plot_results(model, X, y, X_test, y_test, y_pred)
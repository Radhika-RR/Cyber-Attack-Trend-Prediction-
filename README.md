# Cyber Attack Trend Prediction

This project predicts cyber attack trends using machine learning models. It includes scripts for data preprocessing, model training, prediction, and evaluation.

## Project Structure
- `config.py`: Configuration settings for the project.
- `evaluate.py`: Script to evaluate model performance.
- `predict.py`: Script to make predictions using trained models.
- `requirements.txt`: List of required Python packages.
- `data/`: Contains the dataset (`Dataset.csv`).
- `models/`: Stores trained model files (e.g., `linear_regression_model.pkl`).
- `utils/`: Utility functions (`helpers.py`).

## Getting Started
1. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
2. **Run prediction**:
   ```powershell
   python predict.py
   ```
3. **Evaluate model**:
   ```powershell
   python evaluate.py
   ```

## Dataset
The dataset is located in the `data/` folder as `Dataset.csv`.

## Model
Trained models are saved in the `models/` directory.

## Utilities
Helper functions are available in `utils/helpers.py`.

## License
This project is for educational purposes.

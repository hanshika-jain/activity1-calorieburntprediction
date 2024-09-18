from sklearn.ensemble import RandomForestRegressor
import joblib

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestRegressor()

    def train_model(self, X_train, y_train):
        """Trains the RandomForestRegressor model."""
        self.model.fit(X_train, y_train)
        print("Model trained successfully.")
        return self.model

    def save_model(self, model, model_path='model.pkl'):
        """Saves the trained model to a file."""
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}.")
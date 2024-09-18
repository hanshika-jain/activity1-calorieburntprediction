from sklearn.metrics import mean_squared_error

class ModelEvaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, X_test, y_test):
        """Evaluates the model's performance."""
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"Model Evaluation - Mean Squared Error: {mse:.2f}")
        return mse
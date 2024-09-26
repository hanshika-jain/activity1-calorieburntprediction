import unittest
import pandas as pd
from src.load_data import DataLoader
from src.preprocessor import Preprocessor
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator
import joblib

class TestCaloriePredictionIntegration(unittest.TestCase):

    def setUp(self):
        # Set up test data that includes both 'male' and 'female' to avoid OneHotEncoder issues
        data = {
            'Age': [25, 30, 22],
            'Height': [175, 160, 180],
            'Weight': [70, 55, 80],
            'Duration': [30, 45, 60],
            'Heart_Rate': [120, 110, 100],
            'Body_Temp': [37.0, 36.5, 38.0],
            'Gender': ['male', 'female', 'male'],  # Include both 'male' and 'female'
            'Calories': [300, 350, 500]
        }
        self.df = pd.DataFrame(data)

        # Create necessary instances for testing
        self.data_loader = DataLoader(file_path='data/caloriesBurnt.csv')  # You can replace this with any real file path or a mock if needed
        self.numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
        self.categorical_features = ['Gender']

    def test_model_training_and_evaluation(self):
        """Test the full flow of loading data, preprocessing, training, and evaluating the model."""

        # Convert the setUp dataframe to match the structure of your training and testing data
        X_train = self.df[self.numerical_features + self.categorical_features]
        y_train = self.df['Calories']

        # Preprocess the data
        preprocessor = Preprocessor(self.numerical_features, self.categorical_features)
        X_train_transformed, _ = preprocessor.preprocess_data(X_train, X_train)  # Use the same data for test as a dummy

        # Train the model
        model_trainer = ModelTrainer()
        model = model_trainer.train_model(X_train_transformed, y_train)

        # Save and load the model
        model_trainer.save_model(model, 'test_model.pkl')
        loaded_model = joblib.load('test_model.pkl')

        # Evaluate the model (using the same data for simplicity in this test)
        evaluator = ModelEvaluator(loaded_model)
        mse = evaluator.evaluate(X_train_transformed, y_train)

        # Assert the evaluation result (for the sake of the test, just ensure mse is a positive float)
        self.assertGreater(mse, 0)

    def test_user_input_prediction(self):
        """Test that the system can correctly predict based on user input."""
        
        # Preprocess user input that includes 'female' category to ensure it works
        user_input = [[28, 170, 68, 40, 115, 37.2, 'female']]  # Sample user input with 'female' category
        user_input_df = pd.DataFrame(user_input, columns=self.numerical_features + self.categorical_features)

        # Preprocess the user input
        preprocessor = Preprocessor(self.numerical_features, self.categorical_features)
        preprocessor.pipeline.fit(self.df[self.numerical_features + self.categorical_features])  # Fit the pipeline on known data
        user_input_transformed = preprocessor.pipeline.transform(user_input_df)

        # Load the saved model for prediction
        loaded_model = joblib.load('test_model.pkl')
        predicted_calories = loaded_model.predict(user_input_transformed)

        # Check if the prediction is a positive number
        self.assertGreater(predicted_calories[0], 0)

if __name__ == '__main__':
    unittest.main()

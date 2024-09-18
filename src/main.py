import pandas as pd
from load_data import DataLoader
from preprocessor import Preprocessor
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
import joblib

def get_user_input():
    """Collects input values from the user to make predictions."""
    print("\nPlease provide the following details for calorie prediction:")

    age = int(input("Age (in years): "))
    height = float(input("Height (in cm): "))
    weight = float(input("Weight (in kg): "))
    duration = float(input("Duration of activity (in minutes): "))
    heart_rate = float(input("Heart Rate (in bpm): "))
    body_temp = float(input("Body Temperature (in °C): "))
    gender = input("Gender (male/female): ").lower()

    return [age, height, weight, duration, heart_rate, body_temp, gender]

def main():
    # Load and split data
    data_loader = DataLoader(file_path='data/caloriesBurnt.csv')
    X_train, X_test, y_train, y_test = data_loader.split_data()

    # Define feature lists
    numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
    categorical_features = ['Gender']

    # Convert DataFrames
    X_train_df = pd.DataFrame(X_train, columns=numerical_features + categorical_features)
    X_test_df = pd.DataFrame(X_test, columns=numerical_features + categorical_features)

    # Preprocess data
    preprocessor = Preprocessor(numerical_features, categorical_features)
    X_train_transformed, X_test_transformed = preprocessor.preprocess_data(X_train_df, X_test_df)

    # Train model
    model_trainer = ModelTrainer()
    model = model_trainer.train_model(X_train_transformed, y_train)
    model_trainer.save_model(model)

    # Evaluate model
    evaluator = ModelEvaluator(model)
    evaluator.evaluate(X_test_transformed, y_test)

    # Load model for prediction
    model = joblib.load('model.pkl')

    # Collect user input for predictions
    user_input = get_user_input()
    user_input_df = pd.DataFrame([user_input], columns=numerical_features + categorical_features)
    user_input_transformed = preprocessor.pipeline.transform(user_input_df)

    # Make prediction
    predicted_calories = model.predict(user_input_transformed)
    print(f"\nEstimated Calories Burnt: {predicted_calories[0]:.2f} kcal")

if __name__ == "__main__":
    main()

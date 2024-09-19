# CALORIE BURNT PREDICTION

## Project Overview:
This project is designed to predict the number of calories burnt during physical activity based on user-specific details such as age, height, weight, gender, duration of the activity, heart rate, and body temperature. The project leverages a machine learning model trained on a dataset of user activity to make these predictions.

## Features:
`-**Data Preprocessing:** Preprocesses the input data by handling both numerical and categorical variables.`
`-**Model Training and Evaluation:** Trains a machine learning model to predict calorie burn based on the input features. Evaluates the trained model to ensure accuracy and performance.`
`-**User Input for Predictions:** Allows the user to input their details for real-time calorie burn predictions.`

## Project Structure
`calorie_burnt_prediction/`
`│`
`├── data/                    # Contains the dataset (e.g., caloriesBurnt.csv)`
`├── src/                     # Main source code folder`
`│   ├── load_data.py         # Script for loading and splitting data`
`│   ├── preprocessor.py      # Script for data preprocessing`
`│   ├── model_trainer.py     # Script for training and saving the model`
`│   ├── model_evaluator.py   # Script for model evaluation`
`│   ├── main.py              # Main script to run the project`
`│`
`├── README.md                # Project documentation`
`├── requirements.txt         # Python dependencies`

## Prerequisites
`-Python 3.11+`
`-Libraries (as listed in requirements.txt):`
`-pandas`
`-scikit-learn`
`-joblib`
etc.

## Team Members
`-**Member 1 (Mini John):** Responsible for data loading and preprocessing. Handles the loading of datasets and prepares them for model training.`
`-**Member 2 (Shravani Avalkar):** Manages model training and evaluation. Implements the algorithms and evaluates the model's performance.`
`-**Member 3 (Hanshika Jain)**: Oversees the main execution of the code. Manages the `main.py` file that integrates all components and handles predictions. Also responsible for updating the README and maintaining requirements.`
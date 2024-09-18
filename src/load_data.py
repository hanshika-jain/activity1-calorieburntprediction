#load_data file
import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    def _init_(self, file_path):
        self.file_path = file_path
    
    def load_data(self):
        """Loads data from the CSV file."""
        data = pd.read_csv(self.file_path)
        return data
    
    def split_data(self):
        """Splits the data into features and target, then into training and testing sets."""
        data = self.load_data()
        
        if 'User_ID' in data.columns:
            data = data.drop('User_ID', axis=1)
        
        required_columns = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Gender', 'Calories']
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in data: {', '.join(missing_columns)}")
        
        X = data.drop('Calories', axis=1)  # Features
        y = data['Calories']  # Target
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
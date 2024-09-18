#preprocessor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd

class Preprocessor:
    def _init_(self, numerical_features, categorical_features):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(), categorical_features)
            ],
            remainder='passthrough'
        )
    
    def preprocess_data(self, X_train, X_test):
        """Applies the preprocessing pipeline to the data."""
        if not isinstance(X_train, pd.DataFrame) or not isinstance(X_test, pd.DataFrame):
            raise ValueError("Input data must be in DataFrame format.")
        
        X_train_transformed = self.pipeline.fit_transform(X_train)
        X_test_transformed = self.pipeline.transform(X_test)
        return X_train_transformed, X_test_transformed
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

def load_and_preprocess_data(filepath):
    # Load dataset
    data = pd.read_csv(filepath)
    
    # Preprocess data (assuming required preprocessing steps like handling
    # missing values, encoding categorical variables, scaling features, etc.)
    data = data.dropna()  # Simple dropna approach for missing data
    
    # Feature engineering and selection can be done here
    features = data[['soil_moisture', 'temperature', 'rainfall']]
    target = data['crop_yield']
    
    return train_test_split(features, target, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    # Instantiate the model
    model = LinearRegression()
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate RMSE
    rmse = mean_squared_error(y_test, predictions, squared=False)
    print(f"Model RMSE: {rmse}")
    
    return rmse

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def main(data_filepath):
    # Load Data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_filepath)
    
    # Train
    model = train_model(X_train, y_train)
    
    # Evaluate
    evaluate_model(model, X_test, y_test)
    
    # Save the model
    save_model(model, 'crop_yield_model.pkl')

if __name__ == "__main__":
    main('crop_data.csv')

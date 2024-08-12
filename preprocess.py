import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_file(input_file, output_file):
    # Load the data
    data = pd.read_csv(input_file)
    
    # Check if log_level column exists for mapping
    if 'log_level' in data.columns:
        log_level_mapping = {'INFO': 0, 'WARNING': 1, 'ERROR': 2, 'CRITICAL': 3}
        data['log_level'] = data['log_level'].map(log_level_mapping)
    
    # Handle missing values and scale numeric columns
    numerical_columns = ['value', 'session_length', 'num_failed_attempts', 'packet_size', 'num_retries']
    if set(numerical_columns).issubset(data.columns):
        scaler = StandardScaler()
        imputer = SimpleImputer(strategy='mean')
        data[numerical_columns] = scaler.fit_transform(imputer.fit_transform(data[numerical_columns]))
    
    # One-hot encode categorical columns
    categorical_columns = ['message', 'protocol', 'action']
    if set(categorical_columns).issubset(data.columns):
        data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    
    # Handle datetime features
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['hour'] = data['timestamp'].dt.hour
        data['day'] = data['timestamp'].dt.dayofweek
        data = data.drop(columns=['timestamp'])
    
    # Save the preprocessed data
    data.to_csv(output_file, index=False)
    print(f"Processed {input_file} and saved to {output_file}")

def preprocess_data():
    # Define the directories
    log_data_dir = 'data/logs'
    network_data_dir = 'data/network'
    output_dir = 'data/processed'
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process log files
    for file in os.listdir(log_data_dir):
        if file.endswith('.csv'):
            preprocess_file(os.path.join(log_data_dir, file), os.path.join(output_dir, f'processed_{file}'))
    
    # Process network files
    for file in os.listdir(network_data_dir):
        if file.endswith('.csv'):
            preprocess_file(os.path.join(network_data_dir, file), os.path.join(output_dir, f'processed_{file}'))

if __name__ == "__main__":
    preprocess_data()


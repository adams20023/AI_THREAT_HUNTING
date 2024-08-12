import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Directories
processed_data_dir = 'data/processed/'
model_dir = 'models/'

def load_data():
    data_frames = []
    for file in os.listdir(processed_data_dir):
        df = pd.read_csv(os.path.join(processed_data_dir, file))
        data_frames.append(df)
    return pd.concat(data_frames, axis=0)

def retrain_model():
    data = load_data()
    X = data.drop('label', axis=1)  # Assuming there's a 'label' column for target
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model_path = os.path.join(model_dir, 'model.pkl')
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Loaded existing model.")
    else:
        model = RandomForestClassifier(n_estimators=100)
    
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    
    joblib.dump(model, model_path)
    print(f'Model updated and saved to {model_path}')

if __name__ == "__main__":
    retrain_model()


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import joblib

def preprocess_data(X):
    # Check for NaN values
    if X.isnull().values.any():
        print("Missing values found. Imputing missing values...")
    
    # Impute missing values
    numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Imputation and scaling pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), numerical_columns),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_columns)
        ]
    )
    
    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)
    
    # Convert to DataFrame for easier analysis
    X = pd.DataFrame(X_processed)
    
    return X

def train_model():
    # Load your dataset
    data = pd.read_csv('data/processed/processed_synthetic_logs.csv')
    y = data['label']
    X = data.drop(columns=['label'])

    # Preprocess data
    print(f"Columns in X before preprocessing: {X.columns.tolist()}")
    X = preprocess_data(X)  # This will handle missing values as well
    
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
    print(f"Unique values in y: {np.unique(y)}")

    # SMOTE for handling class imbalance
    smote = SMOTE()
    try:
        X_resampled, y_resampled = smote.fit_resample(X, y)
    except ValueError as e:
        print(f"Error during SMOTE resampling: {e}")
        print("Attempting to troubleshoot by ensuring y is properly formatted...")
        X = preprocess_data(X)
        X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    # LightGBM Model
    model = lgb.LGBMClassifier()

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_data_in_leaf': [10, 20],
        'learning_rate': [0.05, 0.1]
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

    try:
        grid_search.fit(X_train, y_train)
    except KeyboardInterrupt:
        print("Training interrupted.")
        return
    except Exception as e:
        print(f"Training failed with exception: {e}")
        return
    
    # Best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters found: {grid_search.best_params_}")
    
    # Predictions and Evaluation
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("ROC AUC score:")
    print(roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]))

    # Feature Importance
    print("Feature importance:")
    lgb.plot_importance(best_model, max_num_features=10)
    
    # Save model
    joblib.dump(best_model, 'models/model.pkl')
    print("Model saved to models/model.pkl")

if __name__ == "__main__":
    train_model()


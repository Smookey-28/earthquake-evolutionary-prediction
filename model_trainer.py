import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

def bin_magnitude(y):
    """
    Bins continuous earthquake magnitudes into categorical classes:
    - Low: < 4.0 (Rarely causes damage, but felt)
    - Medium: 4.0 <= Mag < 5.5 (Minor damage, felt by everyone)
    - High: >= 5.5 (Moderate to severe damage potential)
    """
    bins = [-float('inf'), 4.0, 5.5, float('inf')]
    labels = ['Low', 'Medium', 'High']
    return pd.cut(y, bins=bins, labels=labels)

def train_and_evaluate(df, target_col='magnitude', selected_features=None, test_size=0.2, random_state=42):
    """
    Trains a Random Forest Regressor on the provided features and evaluates both
    regression and classification (after binning) metrics.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # If feature subset is specified, restrict X
    if selected_features is not None:
        X = X[selected_features]
        
    feature_names = list(X.columns)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Train Random Forest Regressor (with full hyperparameters for the final model)
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=4,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Predict continuous magnitudes
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Regression Metrics
    regression_metrics = {
        'train_mse': mean_squared_error(y_train, y_pred_train),
        'test_mse': mean_squared_error(y_test, y_pred_test),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test)
    }
    
    # Classification Metrics (After Binning)
    y_test_binned = bin_magnitude(y_test)
    y_pred_test_binned = bin_magnitude(y_pred_test)
    
    # Get labels order
    labels = ['Low', 'Medium', 'High']
    
    # Confusion Matrix
    cm = confusion_matrix(y_test_binned, y_pred_test_binned, labels=labels)
    
    # Precision, recall, f1, accuracy
    acc = accuracy_score(y_test_binned, y_pred_test_binned)
    f1 = f1_score(y_test_binned, y_pred_test_binned, average='weighted')
    report = classification_report(y_test_binned, y_pred_test_binned, labels=labels, output_dict=True)
    
    classification_metrics = {
        'accuracy': acc,
        'f1_weighted': f1,
        'confusion_matrix': cm,
        'report': report,
        'labels': labels
    }
    
    # Feature Importances
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    results = {
        'model': model,
        'features': feature_names,
        'regression_metrics': regression_metrics,
        'classification_metrics': classification_metrics,
        'feature_importances': feature_importance_df,
        'predictions': {
            'y_true': y_test,
            'y_pred': y_pred_test,
            'y_true_binned': y_test_binned,
            'y_pred_binned': y_pred_test_binned
        }
    }
    
    return results

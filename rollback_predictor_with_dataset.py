import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, jsonify
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_synthetic_dataset(n_samples=10000):
    np.random.seed(42)
    data = {
        'lines_added': np.random.randint(10, 2000, n_samples),
        'lines_deleted': np.random.randint(0, 1000, n_samples),
        'commit_count': np.random.randint(1, 100, n_samples),
        'error_rate': np.random.uniform(0, 0.3, n_samples),
        'latency_ms': np.random.uniform(50, 1000, n_samples),
        'latency_increase': np.random.uniform(0, 1.0, n_samples),
        'test_coverage': np.random.uniform(0, 1.0, n_samples),
        'bug_count': np.random.randint(0, 50, n_samples),
        'deploy_type': np.random.choice(['major', 'minor', 'patch'], n_samples, p=[0.2, 0.5, 0.3]),
        'env': np.random.choice(['prod', 'staging'], n_samples, p=[0.7, 0.3]),
        'rollback_needed': np.zeros(n_samples, dtype=int)
    }
    df = pd.DataFrame(data)
    df['code_churn'] = df['lines_added'] + df['lines_deleted']
    
    for i in range(n_samples):
        prob = 0.15
        if df.loc[i, 'error_rate'] > 0.06:
            prob += 0.45
        if df.loc[i, 'latency_increase'] > 0.2:
            prob += 0.4
        if df.loc[i, 'lines_added'] > 400:
            prob += 0.3
        if df.loc[i, 'code_churn'] > 800:
            prob += 0.25
        if df.loc[i, 'test_coverage'] < 0.8:
            prob += 0.35
        if df.loc[i, 'bug_count'] > 8:
            prob += 0.3
        if df.loc[i, 'deploy_type'] == 'major':
            prob += 0.2
        if df.loc[i, 'env'] == 'prod':
            prob += 0.15
        prob = min(prob, 0.8)
        df.loc[i, 'rollback_needed'] = np.random.choice([0, 1], p=[1-prob, prob])
    
    logging.info(f"Generated dataset with {n_samples} samples, rollback distribution: {df['rollback_needed'].value_counts(normalize=True).to_dict()}")
    return df

def preprocess_data(df):
    df = df.copy()
    le = LabelEncoder()
    df['deploy_type'] = le.fit_transform(df['deploy_type'].astype(str))
    df['env'] = le.fit_transform(df['env'].astype(str))
    return df

def train_model(df):
    X = df.drop('rollback_needed', axis=1)
    y = df['rollback_needed']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logging.info(f"Class distribution: {np.bincount(y_train)}")
    
    param_grid = {
        'n_estimators': [200, 300, 400],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', None]
    }
    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = RandomizedSearchCV(model, param_grid, n_iter=20, cv=3, scoring='roc_auc', n_jobs=-1, random_state=42)
    
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    logging.info(f"Best parameters: {grid_search.best_params_}")
    
    feature_importance = model.feature_importances_
    feature_names = X.columns
    importance_dict = dict(zip(feature_names, feature_importance))
    logging.info("Feature Importance:\n" + "\n".join([f"{k}: {v:.4f}" for k, v in importance_dict.items()]))
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info("Classification Report:\n" + classification_report(y_test, y_pred))
    logging.info(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    return model

def decide_rollback(probability, threshold=0.6):
    risk_level = "low"
    if probability > 0.8:
        risk_level = "high"
    elif probability > 0.5:
        risk_level = "medium"
    if probability > threshold:
        logging.warning(f"High rollback probability: {probability:.2f}")
        return True, risk_level
    return False, risk_level

def execute_rollback(deployment_id):
    logging.info(f"Executing rollback for deployment {deployment_id}")
    return {"status": "Rollback initiated", "deployment_id": deployment_id}

app = Flask(__name__)
model = None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    deployment_id = data.get('deployment_id', 'unknown')
    features = pd.DataFrame([{
        'lines_added': data.get('lines_added', 0),
        'lines_deleted': data.get('lines_deleted', 0),
        'commit_count': data.get('commit_count', 0),
        'error_rate': data.get('error_rate', 0),
        'latency_ms': data.get('latency_ms', 0),
        'latency_increase': data.get('latency_increase', 0),
        'test_coverage': data.get('test_coverage', 0.8),
        'bug_count': data.get('bug_count', 0),
        'code_churn': data.get('lines_added', 0) + data.get('lines_deleted', 0),
        'deploy_type': data.get('deploy_type', 'minor'),
        'env': data.get('env', 'staging')
    }])
    features = preprocess_data(features)
    
    probability = model.predict_proba(features)[:, 1][0]
    
    rollback_needed, risk_level = decide_rollback(probability)
    
    if rollback_needed:
        rollback_result = execute_rollback(deployment_id)
        return jsonify({
            'deployment_id': deployment_id,
            'rollback_probability': float(probability),
            'rollback_needed': rollback_needed,
            'risk_level': risk_level,
            'rollback_result': rollback_result
        })
    else:
        return jsonify({
            'deployment_id': deployment_id,
            'rollback_probability': float(probability),
            'rollback_needed': rollback_needed,
            'risk_level': risk_level
        })

if __name__ == '__main__':
    df = create_synthetic_dataset()
    df = preprocess_data(df)
    
    df.to_csv('synthetic_deployment_data.csv', index=False)
    logging.info("Synthetic dataset saved to 'synthetic_deployment_data.csv'")
    
    model = train_model(df)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
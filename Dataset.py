import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (confusion_matrix, classification_report,
                             RocCurveDisplay, precision_recall_curve, auc)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
# import shap
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from scipy.stats import uniform, randint


def load_and_preprocess(path):
    # 1) Load data
    df = pd.read_csv(path)

    # 2) Enhanced timestamp features
    df['Timestamp'] = pd.to_datetime(
        df['Timestamp'], format='%m/%d/%Y %H:%M', errors='coerce'
    )
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(np.int32)
    df['TimeOfDay'] = pd.cut(df['Hour'],
                             bins=[0, 6, 12, 18, 24],
                             labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    df.drop('Timestamp', axis=1, inplace=True)

    # 3) Drop identifiers
    df.drop(['Transaction_ID', 'User_ID', 'IP_Address_Flag'], axis=1, inplace=True)

    # 4) Separate features and target early to avoid leakage
    X = df.drop('Fraud_Label', axis=1)
    y = df['Fraud_Label'].astype(np.int32)

    return X, y


def create_preprocessor(num_cols, cat_cols):
    # Create separate preprocessing pipelines for numeric and categorical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ])

    return preprocessor


def plot_evaluation_metrics(y_true, y_pred, y_proba):
    # Confusion Matrix
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Non-Fraud', 'Fraud'],
        yticklabels=['Non-Fraud', 'Fraud']
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    # ROC Curve
    plt.subplot(1, 3, 2)
    RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.title('ROC Curve')

    # Precision-Recall Curve
    plt.subplot(1, 3, 3)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (AUC={:.2f})'.format(auc(recall, precision)))

    plt.tight_layout()
    plt.show()


def main():
    # Load & preprocess using uploaded file path
    X, y = load_and_preprocess('dataset.csv')

    # Identify feature types
    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns

    # Create preprocessing pipeline
    preprocessor = create_preprocessor(num_cols, cat_cols)

    # Calculate class weight
    fraud_ratio = np.sum(y == 0) / np.sum(y == 1)

    # Create final pipeline with preprocessing and model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            eval_metric='logloss',
            scale_pos_weight=fraud_ratio,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Hyperparameter search space
    param_dist = {
        'classifier__n_estimators': randint(50, 200),
        'classifier__max_depth': randint(3, 10),
        'classifier__learning_rate': uniform(0.01, 0.3),
        'classifier__subsample': uniform(0.6, 0.4),
        'classifier__colsample_bytree': uniform(0.6, 0.4)
    }

    # Randomized search with cross-validation
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=20,
        cv=StratifiedKFold(n_splits=5),
        scoring='f1',
        random_state=42,
        n_jobs=-1
    )

    # Train model with hyperparameter tuning
    search.fit(X, y)

    # Best model
    best_model = search.best_estimator_

    # Cross-validation scores
    cv_scores = cross_val_score(
        best_model, X, y,
        cv=StratifiedKFold(n_splits=5),
        scoring='f1',
        n_jobs=-1
    )

    print(f"Best parameters: {search.best_params_}")
    print(f"Cross-validation F1 scores: {cv_scores}")
    print(f"Mean CV F1 score: {np.mean(cv_scores):.4f}")

    # Final evaluation on holdout set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))

    # Plot evaluation metrics
    plot_evaluation_metrics(y_test, y_pred, y_proba)

    # Save model and preprocessing
    joblib.dump(best_model, 'fraud_detection_model.pkl')
    print("Model saved as fraud_detection_model.pkl")
if __name__ == '__main__':
    main()
    # XAI: Feature Importance Visualization



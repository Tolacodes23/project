import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (confusion_matrix, classification_report,
                             RocCurveDisplay, precision_recall_curve, auc)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from scipy.stats import uniform, randint
import shap
from imblearn.over_sampling import SMOTE


def load_and_preprocess(path):
    df = pd.read_csv(path)

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

    df.drop(['Transaction_ID', 'User_ID', 'IP_Address_Flag'], axis=1, inplace=True)

    X = df.drop('Fraud_Label', axis=1)
    y = df['Fraud_Label'].astype(np.int32)

    return X, y


def create_preprocessor(num_cols, cat_cols):
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

    plt.subplot(1, 3, 2)
    RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.title('ROC Curve')

    plt.subplot(1, 3, 3)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (AUC={:.2f})'.format(auc(recall, precision)))

    plt.tight_layout()
    plt.show()


def main():
    X, y = load_and_preprocess('dataset.csv')

    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns

    preprocessor = create_preprocessor(num_cols, cat_cols)

    fraud_ratio = np.sum(y == 0) / np.sum(y == 1)

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            eval_metric='logloss',
            scale_pos_weight=fraud_ratio,
            random_state=42,
            n_jobs=-1
        ))
    ])

    param_dist = {
        'classifier__n_estimators': randint(50, 200),
        'classifier__max_depth': randint(3, 10),
        'classifier__learning_rate': uniform(0.01, 0.3),
        'classifier__subsample': uniform(0.6, 0.4),
        'classifier__colsample_bytree': uniform(0.6, 0.4)
    }

    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=20,
        cv=StratifiedKFold(n_splits=5),
        scoring='f1',
        random_state=42,
        n_jobs=-1
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    print(f"Best parameters: {search.best_params_}")

    # Apply preprocessing manually
    X_train_transformed = best_model.named_steps['preprocessor'].fit_transform(X_train)
    X_test_transformed = best_model.named_steps['preprocessor'].transform(X_test)

    print("Before SMOTE:", np.bincount(y_train))
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_transformed, y_train)
    print("After SMOTE:", np.bincount(y_resampled))

    classifier = best_model.named_steps['classifier']
    classifier.fit(X_resampled, y_resampled)

    y_pred = classifier.predict(X_test_transformed)
    y_proba = classifier.predict_proba(X_test_transformed)[:, 1]

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))

    plot_evaluation_metrics(y_test, y_pred, y_proba)

    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    importance_dict = classifier.get_booster().get_score(importance_type='weight')
    importance_df = pd.DataFrame({
        'Feature': [feature_names[int(k[1:])] for k in importance_dict.keys()],
        'FScore': list(importance_dict.values())
    }).sort_values(by='FScore', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df.head(10), x='FScore', y='Feature')
    plt.title('Top 10 Features by FScore (XGBoost)')
    plt.tight_layout()
    plt.show()

    explainer = shap.Explainer(classifier)
    shap_values = explainer(X_test_transformed)
    shap.summary_plot(shap_values, features=X_test_transformed, feature_names=feature_names)

    joblib.dump(classifier, 'fraud_detection_model.pkl')
    print("Model saved as fraud_detection_model.pkl")


if __name__ == '__main__':
    main()

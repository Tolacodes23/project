import pandas as pd
#import numpy as np
import warnings
warnings.filterwarnings('ignore')

#from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
#from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
#import xgboost as xgb
import seaborn as sns
#import shap
import matplotlib.pyplot as plt
#from imblearn.combine import SMOTETomek

# --- Load and Clean Dataset ---
df = pd.read_csv('dataset.csv')
pd.set_option('display.max_columns', None)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Basic overview
print("\n🧾 Head of Dataset:")
print(df.head(5))

print("\n📊 Class Distribution:")
print(df.Fraud_Label.value_counts())

print("\n🧼 Missing Values:")
print(df.isna().sum())

print("\n📎 Duplicate Rows:", df.duplicated().sum())

# --- Convert Timestamp ---
df['Timestamp'] = (
    pd.to_datetime(df['Timestamp']))

# Extract time-based features
df['Hour'] = df['Timestamp'].dt.hour
df['Day'] = df['Timestamp'].dt.day
df['Month'] = df['Timestamp'].dt.month
df['Weekday'] = df['Timestamp'].dt.weekday
df.drop(columns=['Timestamp'], inplace=True)

# --- Encode Categorical Columns ---
categorical_cols = df.select_dtypes(include='object').columns.tolist()
categorical_cols = [col for col in categorical_cols if col not in ['Transaction_ID', 'User_ID']]

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# --- Scale Numeric Features ---
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols.remove('Fraud_Label')  # Don't scale the target label

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\n✅ Preprocessing complete. Shape:", df.shape)
print(df.head())

# --- ✅ Correlation Matrix (numeric only) ---
numeric_df = df.select_dtypes(include=['int64', 'float64'])
cm = numeric_df.corr()

print("\n🔗 Correlation Matrix:")
print(cm)

# --- 🔥 Correlation Heatmap ---
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

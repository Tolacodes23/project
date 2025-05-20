import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
#text cleaning
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import shap
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer

# ----- Load Dataset -----
df = pd.read_csv('dataset.csv')
pd.set_option('display.max_columns', None)
print(df.head(5))
pd.set_option('display.max_colwidth', None)

# ----- Basic EDA -----
print(df.Fraud_Label.value_counts())
print(df.isna().sum())
print(df.duplicated().sum())

# Column name cleanup and drop duplicates
df.columns = df.columns.str.strip()
df.drop_duplicates(inplace=True)

# Convert timestamp and extract features
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
df['Day'] = df['Timestamp'].dt.day
df['Month'] = df['Timestamp'].dt.month
df['Weekday'] = df['Timestamp'].dt.weekday
df.drop(columns=['Timestamp'], inplace=True)

# Encode categorical columns
categorical_cols = df.select_dtypes(include='object').columns.tolist()
categorical_cols = [col for col in categorical_cols if col not in ['Transaction_ID', 'User_ID']]
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Scale numeric features
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols.remove('Fraud_Label')
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("✅ Preprocessing complete. Shape:", df.shape)

# Correlation heatmap
plt.figure(figsize=(12, 10))
# Calculate correlation only on numeric columns
numeric_df = df.select_dtypes(include=['int64', 'float64'])
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isnull(text):
        return ''
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

text_columns = df.select_dtypes(include='object').columns
for col in text_columns:
    df[col] = df[col].apply(clean_text)
print("✅ Text cleaning complete. Cleaned text columns:", list(text_columns))

# Drop non-contributing columns
cols_to_drop = ['Transaction_ID', 'User_ID', 'Timestamp']
cols_to_drop = [col for col in cols_to_drop if col in df.columns]
df.drop(columns=cols_to_drop, inplace=True)
constant_cols = df.columns[df.nunique() <= 1].tolist()
df.drop(columns=constant_cols, inplace=True)
print(f"✅ Dropped non-contributing columns: {cols_to_drop + constant_cols}")

# TF-IDF vectorization
text_col = 'Description'
if text_col in df.columns:
    tfidf = TfidfVectorizer(max_features=100)
    tfidf_matrix = tfidf.fit_transform(df[text_col])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
    df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    df.drop(columns=[text_col], inplace=True)
    print(f"✅ Extracted TF-IDF features from '{text_col}'")

if 'Transaction_Amount' in df.columns and 'Hour' in df.columns:
    df['Amount_Hour'] = df['Transaction_Amount'] * df['Hour']
    print("✅ Created interaction feature 'Amount_Hour'")

# Feature selection using XGBoost
X = df.drop('Fraud_Label', axis=1)
y = df['Fraud_Label']
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X, y)
importances = xgb_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
top_features = importance_df.sort_values(by='Importance', ascending=False).head(20)['Feature'].tolist()
df = df[top_features + ['Fraud_Label']]
print("✅ Selected top 20 important features.")

# Split data
X = df.drop('Fraud_Label', axis=1)
y = df['Fraud_Label']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("✅ Data split complete:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)
print("Original class distribution:\n", y_train.value_counts())

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("\n✅ After applying SMOTE:")
print("X_train_res shape:", X_train_res.shape)
print("y_train_res shape:", y_train_res.shape)
print("Resampled class distribution:\n", y_train_res.value_counts())


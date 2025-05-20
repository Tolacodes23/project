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
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import xgboost as xgb
import seaborn as sns
import shap
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# ----- Load Dataset -----
df = pd.read_csv('dataset.csv')
pd.set_option('display.max_columns', None)
print(df.head(5))
pd.set_option('display.max_colwidth', None)

# ----- Basic EDA -----
g = df.Fraud_Label.value_counts()
print(g)

c = df.isna().sum()
print(c)

d = df.duplicated().sum()
print(d)

# Check numeric columns (initial stage)
numeric_cols = []
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        numeric_cols.append(col)

dt = df.info()
print(dt)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Drop duplicate rows if any
df.drop_duplicates(inplace=True)

# Re-load dataset (you may skip this if not needed)
df = pd.read_csv('dataset.csv')

# ----- Step 1: Clean Column Names -----
df.columns = df.columns.str.strip()

# ----- Step 2: Convert Timestamp -----
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Extract time features (optional but useful for modeling)
df['Hour'] = df['Timestamp'].dt.hour
df['Day'] = df['Timestamp'].dt.day
df['Month'] = df['Timestamp'].dt.month
df['Weekday'] = df['Timestamp'].dt.weekday

# Drop original timestamp if not needed
df.drop(columns=['Timestamp'], inplace=True)

# ----- Step 3: Encode Categorical Columns -----
# List of object-type categorical features
categorical_cols = df.select_dtypes(include='object').columns.tolist()

# Drop identifiers (optional)
categorical_cols = [col for col in categorical_cols if col not in ['Transaction_ID', 'User_ID']]

# Apply Label Encoding (can also use OneHotEncoder if needed)
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# ----- Step 4: Scale Numeric Features -----
# Identify numeric features to scale (excluding label)
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols.remove('Fraud_Label')  # Exclude target label

# Scale features
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# ----- Final Output -----
print("✅ Preprocessing complete. Shape:", df.shape)
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
plt.savefig('correlation_heatmap.png')  # Save the heatmap as an image
plt.close()  # Close the plot to free memory

# Download NLTK data (only needs to run once)
try:
    nltk.download('stopwords')
    nltk.download('wordnet')
except:
    print("NLTK download failed or already downloaded")

# Set up tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define text cleaning function
def clean_text(text):
    if pd.isnull(text):
        return ''
    text = str(text).lower()  # Ensure text is string
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize
    return ' '.join(tokens)

# 🔍 Identify object-type columns (text fields)
text_columns = df.select_dtypes(include='object').columns

# 🧼 Apply text cleaning to each text column
for col in text_columns:
    df[col] = df[col].apply(clean_text)

print("\n✅ Text cleaning complete. Cleaned text columns:", list(text_columns))
print(df.select_dtypes(include='object').columns.tolist())

# ---- DROP NON-CONTRIBUTING COLUMNS ----
# Drop IDs or unique identifiers (adjust based on your dataset)
cols_to_drop = ['Transaction_ID', 'User_ID']  # only if they exist

# Only drop columns that actually exist in df
cols_to_drop = [col for col in cols_to_drop if col in df.columns]
df.drop(columns=cols_to_drop, inplace=True)

# Drop constant columns (no variation)
nunique = df.nunique()
constant_cols = nunique[nunique == 1].index.tolist()
df.drop(columns=constant_cols, inplace=True)

print(f"✅ Dropped non-contributing columns: {cols_to_drop + constant_cols}")

# Feature extraction from text columns if present
from sklearn.feature_extraction.text import TfidfVectorizer

# Check if 'Description' exists in your dataframe
text_col = 'Description'
if text_col in df.columns:
    tfidf = TfidfVectorizer(max_features=100)
    tfidf_matrix = tfidf.fit_transform(df[text_col])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
    df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    df.drop(columns=[text_col], inplace=True)
    print(f"✅ Extracted TF-IDF features from '{text_col}'")

# Example: create interaction feature
if 'Transaction_Amount' in df.columns and 'Hour' in df.columns:
    df['Amount_Hour'] = df['Transaction_Amount'] * df['Hour']
    print("✅ Created interaction feature 'Amount_Hour'")

print("👉 Starting feature selection...")
    
# Define features and target
X = df.drop('Fraud_Label', axis=1)
y = df['Fraud_Label']

# Train a quick model for feature selection
print("Training initial model for feature selection...")
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X, y)

# Get feature importances
importances = xgb_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort and select top features (e.g., top 20)
top_features = importance_df.sort_values(by='Importance', ascending=False).head(20)['Feature'].tolist()
df = df[top_features + ['Fraud_Label']]

print("✅ Selected top 20 important features based on XGBoost.")
print(importance_df.sort_values(by='Importance', ascending=False).head(20))

# Split the data
print("\n👉 Splitting the dataset...")
X = df.drop('Fraud_Label', axis=1)
y = df['Fraud_Label']

# Split dataset into 80% training and 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("✅ Data split complete:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)
print("Original class distribution:\n", y_train.value_counts())

# Apply SMOTE to handle data imbalance
print("\n👉 Applying SMOTE for class balance...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("✅ After applying SMOTE:")
print("X_train_res shape:", X_train_res.shape)
print("y_train_res shape:", y_train_res.shape)
print("Resampled class distribution:\n", y_train_res.value_counts())


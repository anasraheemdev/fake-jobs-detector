import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(df):
    """Preprocess the dataset by handling missing values."""
    df_processed = df.copy()
    
    # Fill missing text with empty string
    text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    for col in text_columns:
        df_processed[col] = df_processed[col].fillna('')
    
    # Fill missing categorical with 'Unknown'
    categorical_columns = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
    for col in categorical_columns:
        df_processed[col] = df_processed[col].fillna('Unknown')
    
    # Fill other missing values
    df_processed['location'] = df_processed['location'].fillna('Unknown')
    df_processed['department'] = df_processed['department'].fillna('Unknown')
    df_processed['salary_range'] = df_processed['salary_range'].fillna('Unknown')
    
    return df_processed

def engineer_features(df_processed):
    """Create engineered features for the model."""
    label_encoders = {}
    
    # Text length features
    df_processed['title_length'] = df_processed['title'].str.len()
    df_processed['company_profile_length'] = df_processed['company_profile'].str.len()
    df_processed['description_length'] = df_processed['description'].str.len()
    df_processed['requirements_length'] = df_processed['requirements'].str.len()
    df_processed['benefits_length'] = df_processed['benefits'].str.len()
    
    # Combined text features
    df_processed['total_text_length'] = (df_processed['title_length'] + 
                                       df_processed['company_profile_length'] + 
                                       df_processed['description_length'] + 
                                       df_processed['requirements_length'] + 
                                       df_processed['benefits_length'])
    
    # Word count features
    df_processed['title_word_count'] = df_processed['title'].str.split().str.len()
    df_processed['description_word_count'] = df_processed['description'].str.split().str.len()
    df_processed['requirements_word_count'] = df_processed['requirements'].str.split().str.len()
    
    # URL count in text
    df_processed['url_count'] = df_processed['description'].str.count('http') + \
                               df_processed['requirements'].str.count('http') + \
                               df_processed['benefits'].str.count('http')
    
    # Email count in text
    df_processed['email_count'] = df_processed['description'].str.count('@') + \
                                 df_processed['requirements'].str.count('@') + \
                                 df_processed['benefits'].str.count('@')
    
    # Salary-related features
    df_processed['has_salary_range'] = (df_processed['salary_range'] != 'Unknown').astype(int)
    df_processed['salary_range_length'] = df_processed['salary_range'].str.len()
    
    # Extract salary information
    df_processed['has_high_salary_keywords'] = (
        df_processed['salary_range'].str.lower().str.contains('k|000|million', na=False) |
        df_processed['description'].str.lower().str.contains('high salary|competitive salary|excellent pay', na=False)
    ).astype(int)
    
    # Check for unrealistic salary promises
    df_processed['has_unrealistic_salary'] = (
        df_processed['salary_range'].str.lower().str.contains('per week.*[5-9][0-9]{3,}|weekly.*[5-9][0-9]{3,}', na=False, regex=True) |
        df_processed['description'].str.lower().str.contains('earn.*[5-9][0-9]{3,}.*week|make.*[5-9][0-9]{3,}.*week', na=False, regex=True)
    ).astype(int)
    
    # Check for suspicious keywords
    suspicious_keywords = ['urgent', 'immediate', 'quick money', 'work from home', 'earn money', 
                          'make money', 'easy money', 'fast cash', 'quick cash', 'no experience needed']
    
    for keyword in suspicious_keywords:
        df_processed[f'has_{keyword.replace(" ", "_")}'] = (
            df_processed['description'].str.lower().str.contains(keyword, na=False) |
            df_processed['requirements'].str.lower().str.contains(keyword, na=False) |
            df_processed['benefits'].str.lower().str.contains(keyword, na=False)
        ).astype(int)
    
    # Encode categorical variables
    categorical_features = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
    for feature in categorical_features:
        le = LabelEncoder()
        df_processed[f'{feature}_encoded'] = le.fit_transform(df_processed[feature])
        label_encoders[feature] = le
    
    return df_processed, label_encoders

def prepare_features(df_processed):
    """Prepare features for modeling."""
    # Select engineered features
    engineered_features = [
        'title_length', 'company_profile_length', 'description_length', 'requirements_length', 'benefits_length',
        'total_text_length', 'title_word_count', 'description_word_count', 'requirements_word_count',
        'url_count', 'email_count', 'telecommuting', 'has_company_logo', 'has_questions',
        'has_salary_range', 'salary_range_length', 'has_high_salary_keywords', 'has_unrealistic_salary'
    ]
    
    # Add encoded categorical features
    categorical_features = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
    encoded_features = [f'{feature}_encoded' for feature in categorical_features]
    engineered_features.extend(encoded_features)
    
    # Add suspicious keyword features
    suspicious_keywords = ['urgent', 'immediate', 'quick money', 'work from home', 'earn money', 
                          'make money', 'easy money', 'fast cash', 'quick cash', 'no experience needed']
    suspicious_features = [f'has_{keyword.replace(" ", "_")}' for keyword in suspicious_keywords]
    engineered_features.extend(suspicious_features)
    
    # Create TF-IDF features
    df_processed['combined_text'] = (df_processed['title'] + ' ' + 
                                    df_processed['company_profile'] + ' ' + 
                                    df_processed['description'] + ' ' + 
                                    df_processed['requirements'] + ' ' + 
                                    df_processed['benefits'])
    
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
    tfidf_features = tfidf_vectorizer.fit_transform(df_processed['combined_text'])
    
    # Create feature matrix
    X_engineered = df_processed[engineered_features].values
    X_tfidf = tfidf_features.toarray()
    X_combined = np.hstack([X_engineered, X_tfidf])
    y = df_processed['fraudulent'].values
    
    return X_combined, y, tfidf_vectorizer

def train_and_save_model():
    """Train the model and save it along with preprocessors."""
    print("Loading dataset...")
    
    # Load the dataset
    try:
        df = pd.read_csv('fake_job_postings.csv')
        print(f"Dataset loaded successfully with {len(df)} records")
    except FileNotFoundError:
        print("Error: fake_job_postings.csv not found!")
        print("Please ensure the dataset file is in the same directory as this script.")
        return
    
    print("Preprocessing data...")
    # Preprocess data
    df_processed = preprocess_data(df)
    df_processed, label_encoders = engineer_features(df_processed)
    
    print("Preparing features...")
    # Prepare features
    X_combined, y, tfidf_vectorizer = prepare_features(df_processed)
    
    print("Splitting data...")
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training model...")
    # Train the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # Evaluate the model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model and preprocessors
    print("Saving model and preprocessors...")
    
    model_data = {
        'model': model,
        'tfidf_vectorizer': tfidf_vectorizer,
        'label_encoders': label_encoders,
        'accuracy': accuracy,
        'feature_names': {
            'engineered_features': [
                'title_length', 'company_profile_length', 'description_length', 'requirements_length', 'benefits_length',
                'total_text_length', 'title_word_count', 'description_word_count', 'requirements_word_count',
                'url_count', 'email_count', 'telecommuting', 'has_company_logo', 'has_questions',
                'has_salary_range', 'salary_range_length', 'has_high_salary_keywords', 'has_unrealistic_salary'
            ],
            'categorical_features': ['employment_type', 'required_experience', 'required_education', 'industry', 'function'],
            'suspicious_keywords': ['urgent', 'immediate', 'quick money', 'work from home', 'earn money', 
                                  'make money', 'easy money', 'fast cash', 'quick cash', 'no experience needed']
        }
    }
    
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model saved successfully as 'trained_model.pkl'")
    print(f"Model file size: {round(os.path.getsize('trained_model.pkl') / (1024*1024), 2)} MB")

if __name__ == '__main__':
    import os
    train_and_save_model()
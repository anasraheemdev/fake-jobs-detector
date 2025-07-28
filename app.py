from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables to store the model and preprocessors
model_data = None

def load_trained_model():
    """Load the pre-trained model from pickle file."""
    global model_data
    
    pickle_path = 'trained_model.pkl'
    
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(
            f"Trained model file '{pickle_path}' not found!\n"
            "Please run 'python train_model.py' first to generate the model file."
        )
    
    print("Loading pre-trained model from pickle file...")
    
    try:
        with open(pickle_path, 'rb') as f:
            model_data = pickle.load(f)
        
        print(f"Model loaded successfully!")
        print(f"Model accuracy: {model_data.get('accuracy', 'Unknown'):.4f}")
        print(f"Model file size: {round(os.path.getsize(pickle_path) / (1024*1024), 2)} MB")
        
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

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

def engineer_features_for_prediction(df_processed):
    """Create engineered features for new data prediction."""
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
    suspicious_keywords = model_data['feature_names']['suspicious_keywords']
    
    for keyword in suspicious_keywords:
        df_processed[f'has_{keyword.replace(" ", "_")}'] = (
            df_processed['description'].str.lower().str.contains(keyword, na=False) |
            df_processed['requirements'].str.lower().str.contains(keyword, na=False) |
            df_processed['benefits'].str.lower().str.contains(keyword, na=False)
        ).astype(int)
    
    return df_processed

def predict_job_posting(job_data):
    """Predict if a job posting is fraudulent using the pre-trained model."""
    global model_data
    
    if model_data is None:
        raise Exception("Model not loaded. Please restart the application.")
    
    # Extract components from model_data
    model = model_data['model']
    tfidf_vectorizer = model_data['tfidf_vectorizer']
    label_encoders = model_data['label_encoders']
    feature_names = model_data['feature_names']
    
    # Create DataFrame from job data
    df_job = pd.DataFrame([job_data])
    
    # Preprocess
    df_processed = preprocess_data(df_job)
    df_processed = engineer_features_for_prediction(df_processed)
    
    # Create combined text for TF-IDF
    df_processed['combined_text'] = (df_processed['title'] + ' ' + 
                                    df_processed['company_profile'] + ' ' + 
                                    df_processed['description'] + ' ' + 
                                    df_processed['requirements'] + ' ' + 
                                    df_processed['benefits'])
    
    # Transform with TF-IDF
    tfidf_features = tfidf_vectorizer.transform(df_processed['combined_text'])
    
    # Prepare engineered features
    engineered_features = feature_names['engineered_features']
    
    # Encode categorical features
    categorical_features = feature_names['categorical_features']
    for feature in categorical_features:
        le = label_encoders[feature]
        sample_values = df_processed[feature].values
        encoded_values = []
        for val in sample_values:
            if val in le.classes_:
                encoded_values.append(le.transform([val])[0])
            else:
                # Handle unknown categories with a default value (0)
                encoded_values.append(0)
        df_processed[f'{feature}_encoded'] = encoded_values
    
    encoded_features = [f'{feature}_encoded' for feature in categorical_features]
    engineered_features.extend(encoded_features)
    
    # Add suspicious keyword features
    suspicious_features = [f'has_{keyword.replace(" ", "_")}' for keyword in feature_names['suspicious_keywords']]
    engineered_features.extend(suspicious_features)
    
    # Create feature matrix
    X_engineered = df_processed[engineered_features].values
    X_tfidf = tfidf_features.toarray()
    X_combined = np.hstack([X_engineered, X_tfidf])
    
    # Make prediction
    prediction = model.predict(X_combined)[0]
    prediction_proba = model.predict_proba(X_combined)[0]
    
    # Get feature analysis
    feature_analysis = analyze_features(df_processed.iloc[0])
    
    return {
        'prediction': int(prediction),
        'confidence': float(prediction_proba[1]),  # Probability of being fraudulent
        'confidence_scores': {
            'legitimate': float(prediction_proba[0]),
            'fraudulent': float(prediction_proba[1])
        },
        'is_fraudulent': bool(prediction),
        'feature_analysis': feature_analysis
    }

def analyze_features(job_row):
    """Analyze features to provide insights."""
    analysis = {
        'suspicious_keywords': [],
        'red_flags': [],
        'green_flags': []
    }
    
    # Check suspicious keywords
    suspicious_keywords = model_data['feature_names']['suspicious_keywords']
    
    for keyword in suspicious_keywords:
        feature_name = f'has_{keyword.replace(" ", "_")}'
        if job_row[feature_name] == 1:
            analysis['suspicious_keywords'].append(keyword)
    
    # Check red flags
    if job_row['url_count'] > 0:
        analysis['red_flags'].append(f"Contains {job_row['url_count']} URLs in description")
    
    if job_row['email_count'] > 0:
        analysis['red_flags'].append(f"Contains {job_row['email_count']} email addresses")
    
    if job_row['has_company_logo'] == 0:
        analysis['red_flags'].append("No company logo")
    
    if job_row['company_profile_length'] < 50:
        analysis['red_flags'].append("Very short or missing company profile")
    
    if job_row['description_length'] < 100:
        analysis['red_flags'].append("Very short job description")
    
    if job_row['has_salary_range'] == 0:
        analysis['red_flags'].append("No salary information provided")
    
    if job_row['has_unrealistic_salary'] == 1:
        analysis['red_flags'].append("Unrealistic salary promises (potential scam)")
    
    # Check green flags
    if job_row['has_company_logo'] == 1:
        analysis['green_flags'].append("Company logo present")
    
    if job_row['has_questions'] == 1:
        analysis['green_flags'].append("Application questions present")
    
    if job_row['company_profile_length'] > 200:
        analysis['green_flags'].append("Detailed company profile")
    
    if job_row['description_length'] > 500:
        analysis['green_flags'].append("Detailed job description")
    
    if job_row['requirements_length'] > 100:
        analysis['green_flags'].append("Detailed requirements section")
    
    if job_row['has_salary_range'] == 1:
        analysis['green_flags'].append("Salary information provided")
    
    if job_row['has_high_salary_keywords'] == 1 and job_row['has_unrealistic_salary'] == 0:
        analysis['green_flags'].append("Professional salary terminology used")
    
    return analysis

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy', 
        'model_loaded': model_data is not None,
        'model_accuracy': model_data.get('accuracy', 'Unknown') if model_data else None
    })

@app.route('/model-info')
def model_info():
    """Get model information."""
    if model_data is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'accuracy': model_data.get('accuracy', 'Unknown'),
        'feature_count': {
            'engineered': len(model_data['feature_names']['engineered_features']),
            'categorical': len(model_data['feature_names']['categorical_features']),
            'suspicious_keywords': len(model_data['feature_names']['suspicious_keywords']),
            'tfidf': 1000  # Max features set in TfidfVectorizer
        },
        'model_type': 'RandomForestClassifier'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for prediction."""
    try:
        if model_data is None:
            return jsonify({'error': 'Model not loaded. Please restart the application.'}), 500
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['title', 'description']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Set default values for optional fields
        job_data = {
            'title': data.get('title', ''),
            'location': data.get('location', 'Unknown'),
            'department': data.get('department', 'Unknown'),
            'salary_range': data.get('salary_range', 'Unknown'),
            'company_profile': data.get('company_profile', ''),
            'description': data.get('description', ''),
            'requirements': data.get('requirements', ''),
            'benefits': data.get('benefits', ''),
            'telecommuting': int(data.get('telecommuting', 0)),
            'has_company_logo': int(data.get('has_company_logo', 0)),
            'has_questions': int(data.get('has_questions', 0)),
            'employment_type': data.get('employment_type', 'Unknown'),
            'required_experience': data.get('required_experience', 'Unknown'),
            'required_education': data.get('required_education', 'Unknown'),
            'industry': data.get('industry', 'Unknown'),
            'function': data.get('function', 'Unknown')
        }
        
        # Make prediction
        result = predict_job_posting(job_data)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/examples')
def examples():
    """Return example job postings."""
    examples = [
        {
            'title': 'Marketing Intern',
            'location': 'US, NY, New York',
            'department': 'Marketing',
            'salary_range': '$15-20/hour',
            'company_profile': 'We are a well-established marketing agency with 10+ years of experience in digital marketing and brand development.',
            'description': 'We are looking for a marketing intern to help with social media campaigns, content creation, and market research.',
            'requirements': 'Currently enrolled in marketing or related field. Strong communication skills required. Familiarity with social media platforms.',
            'benefits': 'Flexible hours, mentorship opportunities, potential for full-time position after internship completion.',
            'telecommuting': 0,
            'has_company_logo': 1,
            'has_questions': 1,
            'employment_type': 'Internship',
            'required_experience': 'Not Applicable',
            'required_education': 'Some College Coursework Completed',
            'industry': 'Marketing and Advertising',
            'function': 'Marketing'
        },
        {
            'title': 'URGENT: Work from Home - Make Quick Money',
            'location': 'Remote',
            'department': 'Sales',
            'salary_range': '$5000-$10000 per week',
            'company_profile': '',
            'description': 'URGENT opportunity to make quick money from home! No experience needed. Start earning immediately. Easy money guaranteed!',
            'requirements': 'No experience needed. Just need a computer and internet connection. Make money fast!',
            'benefits': 'Quick cash, easy money, work from anywhere. Fast cash payments.',
            'telecommuting': 1,
            'has_company_logo': 0,
            'has_questions': 0,
            'employment_type': 'Full-time',
            'required_experience': 'Not Applicable',
            'required_education': 'High School or equivalent',
            'industry': 'Sales',
            'function': 'Sales'
        },
        {
            'title': 'Software Engineer',
            'location': 'US, CA, San Francisco',
            'department': 'Engineering',
            'salary_range': '$120,000-$180,000 annually',
            'company_profile': 'Leading technology company focused on innovative solutions in cloud computing and artificial intelligence.',
            'description': 'We are seeking a talented software engineer to join our growing team. You will work on cutting-edge projects and collaborate with experienced developers.',
            'requirements': 'Bachelor\'s degree in Computer Science or related field. 3+ years of experience with Python, JavaScript, and modern web frameworks.',
            'benefits': 'Competitive salary, comprehensive health insurance, 401k matching, flexible work arrangements, professional development budget.',
            'telecommuting': 0,
            'has_company_logo': 1,
            'has_questions': 1,
            'employment_type': 'Full-time',
            'required_experience': 'Mid-Senior level',
            'required_education': 'Bachelor\'s Degree',
            'industry': 'Computer Software',
            'function': 'Engineering'
        }
    ]
    return jsonify(examples)

# Initialize the model when the module is imported
try:
    load_trained_model()
    print("Flask app initialized successfully with pre-trained model!")
except Exception as e:
    print(f"Warning: {str(e)}")
    print("The app will start but predictions will not work until the model is loaded.")

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=8080)
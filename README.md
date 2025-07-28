# 🤖 Fake Job Posting Detector

A machine learning-powered web app to detect fraudulent job postings in real-time. This tool analyzes job descriptions, company profiles, and other features to flag potentially scammy listings.

**GitHub Repository**: [https://github.com/anasraheemdev/fake-jobs-detector](https://github.com/anasraheemdev/fake-jobs-detector)

---

![App Screenshot](https://raw.githubusercontent.com/anasraheemdev/fake-jobs-detector/main/image_d6649d.png)

---

### ✨ Key Features

-   **🔍 Real-Time Analysis**: Get an instant fraud prediction by filling out a simple form.
-   **🧠 Intelligent Detection**: Uses a Random Forest model to analyze text, metadata, and company details.
-   **📊 Detailed Results**: See a clear "Real" or "Fake" prediction along with a confidence score.
-   **🎨 Modern UI**: Clean, responsive, and user-friendly interface.
-   **🧪 Example Loader**: Instantly load sample data to see the detector in action.

### ⚙️ Technology Stack

-   **Backend**: Python, Flask
-   **Machine Learning**: Scikit-learn, Pandas, NumPy
-   **Frontend**: HTML, CSS, JavaScript
-   **Core Algorithm**: Random Forest Classifier with TF-IDF for text processing.

### 🚀 Quick Start Guide

Get the project running on your local machine in a few simple steps.

1.  **Clone the Repo**
    ```bash
    git clone [https://github.com/anasraheemdev/fake-jobs-detector.git](https://github.com/anasraheemdev/fake-jobs-detector.git)
    cd fake-jobs-detector
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *This installs Flask, scikit-learn, and other necessary libraries.*

3.  **Run the Application**
    ```bash
    python app.py
    ```

4.  **View in Browser**
    -   Open your browser and go to `http://127.0.0.1:8080`.
    -   Fill in the form or use the "Load Example" button to test it out!

### 🧠 How the Model Works

The detection logic is powered by a Random Forest model trained on the `fake_job_postings.csv` dataset. The model doesn't just look at words; it analyzes a combination of engineered features:

1.  **Text Content Analysis**:
    -   It uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text from the job description, requirements, and benefits into numerical features. This helps identify unusual or spammy language.
2.  **Metadata Checks**:
    -   **Company Profile**: Does the posting include a company logo? A detailed profile?
    -   **Job Details**: What is the required experience and education level? Is it a telecommuting job?
    -   **Input Lengths**: Very short or very long text fields can be indicative of fraud.
3.  **Keyword Search**:
    -   The model implicitly learns the importance of certain keywords associated with fraudulent posts.

The final prediction is a probability score indicating the likelihood of the posting being fake.

### 📂 Project File Structure

A recommended structure for a project like this.


/
├── app.py                  # The main Flask server
├── random_forest_model.pkl # The pre-trained ML model file
├── model_training.ipynb    # Jupyter Notebook showing how the model was trained
├── requirements.txt        # List of Python packages to install
├── fake_job_postings.csv   # The dataset used for training
├── README.md               # This file
├── LICENSE                 # Project license file
├── templates/
│   └── index.html          # Frontend HTML page
└── static/
├── css/style.css       # Stylesheet
└── js/script.js        # JavaScript for the frontend


### 🤝 Want to Contribute?

Contributions are welcome! If you have ideas for improvements, feel free to fork the repo, make your changes, and submit a pull request.

### ⚠️ Disclaimer

This tool is intended for educational and demonstration purposes. The predictions are based on statistical probabilities and should not be used as the sole factor in judging a job offer. Always do your own research and due diligence.

### 📜 License

This project is distributed under the MIT License. See the `LICENSE` file for more information.

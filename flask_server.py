import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

app = Flask(__name__)

# Base directory for reliable relative path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')  # Point to models/ or BASE_DIR if kept in root

def get_model_path(filename):
    # Checks models/ folder first, falls back to root directory
    in_models = os.path.join(MODELS_DIR, filename)
    return in_models if os.path.exists(in_models) else os.path.join(BASE_DIR, filename)

# Load Machine Learning & Deep Learning artifacts
rf_model = joblib.load(get_model_path('sarcasm_random_forest.joblib'))
tfidf = joblib.load(get_model_path('tfidf_vectorizer.joblib'))

lstm_model = load_model(get_model_path('sarcasm_lstm.keras'))
with open(get_model_path('lstm_tokenizer.json'), 'r', encoding='utf-8') as f:
    tokenizer_data = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_data)

def load_external_data(filename='train-balanced-sarcasm.csv'):
    filepath = os.path.join(BASE_DIR, filename)
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        return None

@app.route('/')
def index():
    return render_template('web.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.get('text')
    if not data:
        return render_template('web.html')

    # Model 1: Random Forest
    tfidf_input = tfidf.transform([data])
    rf_pred = int(rf_model.predict(tfidf_input)[0])

    # Model 2: LSTM
    sequences = tokenizer.texts_to_sequences([data])
    padded = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')
    lstm_prob = float(lstm_model.predict(padded)[0][0])
    lstm_pred = 1 if lstm_prob > 0.5 else 0

    ensemble_data = {
        "prediction": "Sarcastic" if rf_pred == 1 else "Not Sarcastic",
        "description": "Analyzed using TF-IDF features through multiple decision trees."
    }

    confidence_val = lstm_prob if lstm_pred == 1 else (1 - lstm_prob)

    nn_data = {
        "prediction": "Sarcastic" if lstm_pred == 1 else "Not Sarcastic",
        "confidence": f"{round(confidence_val * 100, 2)}%",
        "description": "Analyzed sequential context using a Long Short-Term Memory network."
    }

    return render_template(
        'web.html',
        user_input=data,
        ensemble_result=ensemble_data,
        nn_result=nn_data
    )

@app.route('/view-data')
def view_data():
    df = load_external_data()
    if df is not None:
        return df.head(10).to_html(classes='table table-striped')
    return "CSV file not found. Place 'train-balanced-sarcasm.csv' in the project root."

if __name__ == '__main__':
    # Secure debug flag controlled via environment variable
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=debug_mode)
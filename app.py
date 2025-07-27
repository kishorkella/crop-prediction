from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load trained model and preprocessing objects
model = load_model("crop_prediction_model.h5")

# Create a new StandardScaler with reasonable defaults for agricultural data
scaler = StandardScaler()
default_ranges = np.array([
    [0, 140],    # N
    [0, 140],    # P
    [0, 200],    # K
    [0, 50],     # temperature
    [0, 100],    # humidity
    [0, 14],     # pH
    [0, 300]     # rainfall
])
scaler.mean_ = np.mean(default_ranges, axis=1)
scaler.scale_ = np.std(default_ranges, axis=1)

# Load label encoder or label list
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)  # This is probably a list/array of crop names

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from form
        features = [float(request.form[key]) for key in ["N", "P", "K", "temperature", "humidity", "pH", "rainfall"]]
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Normalize input
        features_scaled = scaler.transform(features_array)
        prediction = model.predict(features_scaled)[0]

        # Get top 3 predictions
        top_3_indices = np.argsort(prediction)[-3:][::-1]
        
        # Updated: label_encoder is a list/array, not an object with inverse_transform()
        top_3_crops = [label_encoder[i] for i in top_3_indices]
        top_3_probabilities = prediction[top_3_indices]

        results = list(zip(top_3_crops, top_3_probabilities))
        
        return render_template('index.html', results=results)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)

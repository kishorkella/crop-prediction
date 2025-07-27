🌾 Crop Recommendation System using ANN

This project is a **web-based Crop Prediction System** built with **Flask** and powered by an **Artificial Neural Network (ANN)** model. It helps farmers and agricultural experts determine the most suitable crops to grow based on soil and environmental conditions.

## 🚀 Features

- Predicts top 3 most suitable crops based on:
  - Nitrogen (N)
  - Phosphorus (P)
  - Potassium (K)
  - Temperature (°C)
  - Humidity (%)
  - pH value of the soil
  - Rainfall (mm)
- Built using a trained **ANN model** in TensorFlow/Keras
- Clean and simple **HTML frontend (index.html)** inside the `public` folder
- Real-time prediction results displayed with probabilities

## 🧠 Model

- The model used is an **Artificial Neural Network (ANN)** trained on agricultural datasets.
- It is saved as a `.h5` file: `crop_prediction_model.h5`.
- Inputs are scaled using `StandardScaler`.
- Output layer returns class probabilities for various crops.
- Top 3 crops are selected and mapped back using a label list (`label_encoder.pkl`).

## 📁 Project Structure

crop-prediction/
│
├── app.py # Flask backend
├── crop_prediction_model.h5 # Trained ANN model
├── label_encoder.pkl # Pickled label list
├── scaler.pkl # (Optional) Scaler object (if saved)
├── public/
│ └── templates/
│ └── index.html # HTML UI for user inputs and results


> Note: In Flask, templates should typically be inside the `templates/` folder. If your `index.html` is in `public/`, make sure your Flask config reflects that.

## 🧪 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/kishorkella/crop-prediction.git
   cd crop-prediction
Create and activate a virtual environment:


python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install the dependencies:


pip install -r requirements.txt
Run the Flask app:

python app.py
Open your browser and go to:

http://127.0.0.1:5000/
🧾 Requirements
Python 3.x

Flask

NumPy

TensorFlow

Scikit-learn

Pickle

You can create a requirements.txt file using:

pip freeze > requirements.txt

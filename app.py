from flask import Flask, request, render_template
import pickle
import numpy as np

# ------------------------------
# Load model and scaler
# ------------------------------
with open('logistic_model.pkl','rb') as f:
    model = pickle.load(f)

with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)

# ------------------------------
# Initialize Flask
# ------------------------------
app = Flask(__name__)

# ------------------------------
# Home page
# ------------------------------
@app.route('/')
def home():
    return render_template('index.html')

# ------------------------------
# Predict route
# ------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)
        
        # Scale features
        final_features_scaled = scaler.transform(final_features)
        
        # Predict
        prediction = model.predict(final_features_scaled)[0]
        probability = model.predict_proba(final_features_scaled)[0][1]
        
        # Result message
        if prediction == 1:
            result = f"Patient is likely Diabetic (Probability: {probability:.2f})"
        else:
            result = f"Patient is likely Non-Diabetic (Probability: {probability:.2f})"
        
        return render_template('index.html', prediction_text=result)
    
    except Exception as e:
        return f"Error: {str(e)}"

# ------------------------------
# Run Flask
# ------------------------------
if __name__ == "__main__":
    app.run(port = 50008,debug=True)

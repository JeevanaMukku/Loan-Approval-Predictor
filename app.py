from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load saved model (must be in the same folder)
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        income = float(request.form['income'])
        loan_amount = float(request.form['loan_amount'])
        credit_score = float(request.form['credit_score'])
        features = np.array([[income, loan_amount, credit_score]])
        pred = model.predict(features)[0]
        result = "✅ Eligible" if pred == 1 else "❌ Not Eligible"
    except Exception:
        result = "Error: please enter valid numbers."
    return render_template('index.html', result=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

from flask import Flask, request, jsonify, render_template # type: ignore
from flask_cors import CORS # type: ignore
import pickle
import numpy as np # type: ignore

app = Flask(__name__)
CORS(app)  # Enable CORS if your frontend and backend run on different domains/ports

# Load your pre-trained models
with open('churn_model_pickle', 'rb') as f:
    churn_model = pickle.load(f)

with open('credit_card_pickle', 'rb') as f:
    allowance_model = pickle.load(f)


@app.route('/')
def index():
    # Render the frontend (index.html) from the templates folder
    return render_template('index.html')


@app.route('/predict_churn', methods=['POST'])
def predict_churn():
    """
    Expects a JSON with the following keys:
    - CustomerId
    - Age
    - CreditScore
    - Tenure
    - TransactionFrequency
    - AvgTransactionAmount
    - ComplaintsFiled
    - CustomerSatisfaction
    - HasLoan   (boolean)
    - Balance
    """
    data = request.get_json(force=True)
    try:
        # Convert input features to float (or the appropriate type) as required by your model.
        features = [
            float(data['Age']),
            float(data['CreditScore']),
            float(data['Tenure']),
            float(data['TransactionFrequency']),
            float(data['AvgTransactionAmount']),
            float(data['ComplaintsFiled']),
            float(data['CustomerSatisfaction']),
            1.0 if data['HasLoan'] else 0.0,
            float(data['Balance'])
        ]
    except Exception as e:
        return jsonify({
            'error': 'Invalid input format for churn prediction.',
            'message': str(e)
        }), 400

    # Reshape features to match model input
    features_array = np.array(features).reshape(1, -1)
    prediction = churn_model.predict(features_array)[0]
    # Assuming your model returns 1 for high churn risk and 0 for low risk
    result = "High chance of churn" if prediction == 1 else "Low chance of churn"

    return jsonify({
        'CustomerId': data.get('CustomerId', None),
        'prediction': result
    })


@app.route('/predict_allowance', methods=['POST'])
def predict_allowance():
    """
    Expects a JSON with the following keys:
    - CreditScore
    - Gender         (string, e.g., "Male", "Female", or "Other")
    - Age
    - Tenure
    - Balance
    - NumOfProducts
    - HasCrCard      (boolean)
    - IsActiveMember (boolean)
    - EstimatedSalary
    """
    data = request.get_json(force=True)
    try:
        # Process gender as needed for your model; here we encode "Male" as 1, others as 0.
        gender_encoded = 1.0 if data['Gender'].lower() == 'male' else 0.0

        features = [
            float(data['CreditScore']),
            gender_encoded,
            float(data['Age']),
            float(data['Tenure']),
            float(data['Balance']),
            float(data['NumOfProducts']),
            1.0 if data['HasCrCard'] else 0.0,
            1.0 if data['IsActiveMember'] else 0.0,
            float(data['EstimatedSalary'])
        ]
    except Exception as e:
        return jsonify({
            'error': 'Invalid input format for allowance prediction.',
            'message': str(e)
        }), 400

    features_array = np.array(features).reshape(1, -1)
    prediction = allowance_model.predict(features_array)[0]
    # Assuming your model returns 1 for allowance approved and 0 for not approved
    result = "Allowance Approved" if prediction == 1 else "Allowance Not Approved"

    return jsonify({
        'prediction': result
    })


if __name__ == '__main__':
    app.run(debug=True)
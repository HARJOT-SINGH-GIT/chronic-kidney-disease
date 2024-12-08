from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)

# Load the pre-trained model
with open('chronic_kidney_disease_ada.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index_view():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract form data
            input_data = [
                float(request.form['age']),
                float(request.form['blood_pressure']),
                float(request.form['specific_gravity']),
                float(request.form['sugar']),
                int(request.form['red_blood_cells']),
                int(request.form['pus_cell']),
                int(request.form['pus_cell_clumps']),
                int(request.form['bacteria']),
                float(request.form['blood_glucose_random']),
                float(request.form['blood_urea']),
                float(request.form['serum_creatinine']),
                float(request.form['sodium']),
                float(request.form['potassium']),
                float(request.form['haemoglobin']),
                float(request.form['packed_cell_volume']),
                float(request.form['white_blood_cell_count']),
                float(request.form['red_blood_cell_count']),
                int(request.form['hypertension']),
                int(request.form['coronary_artery_disease']),
                int(request.form['appetite']),
                int(request.form['peda_edema']),
                int(request.form['aanemia'])
            ]

            # Convert the input data to a numpy array and reshape for the model
            input_array = np.array([input_data])

            # Predict using the model
            prediction = model.predict(input_array)

            # Convert the prediction to a JSON response
            return jsonify({'prediction': prediction.tolist()})

        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return "Invalid request method", 405

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

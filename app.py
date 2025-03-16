from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load dataset for location dropdown
df = pd.read_csv("data/Merged_All_Cities.csv")
unique_locations = sorted(df["Location"].dropna().unique())

# Load trained model and preprocessing objects
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('selected_features.pkl', 'rb') as f:
    selected_features = pickle.load(f)

# Flask App Initialization
app = Flask(__name__, template_folder='.')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/form')
def show_form():
    return render_template('indexhouse.html', locations=unique_locations)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

# Route 8: Submit Feedback
@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    try:
        rating = int(request.form['rating'])
        comment = request.form.get('comment', '')
        conn.execute('INSERT INTO feedback (rating, comment) VALUES (?, ?)', (rating, comment))
        conn.commit()
        return '', 200  # Success response
    except Exception as e:
        print(f"Error saving feedback: {e}")
        return '', 400  # Error response
    
    # Route 9: View Feedback
@app.route('/view_feedback')
def view_feedback():
    feedbacks = conn.execute('SELECT * FROM feedback').fetchall()
    return render_template('view_feedback.html', feedbacks=feedbacks)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()

        # Convert categorical values (Yes/No) to 1/0
        categorical_mappings = {
            "yes": 1,
            "no": 0
        }
        
        # Prepare features dictionary with defaults
        features_dict = {}
        
        # Handle all features from dataset
        for col in selected_features:
            if col == "Location":
                # Encode Location using the trained LabelEncoder
                location_encoder = label_encoders["Location"]
                try:
                    features_dict[col] = location_encoder.transform([data[col]])[0]
                except ValueError:
                    return jsonify({'error': f'Unknown location: {data[col]}'})
            
            elif col in ["Resale", "MaintenanceStaff", "Gymnasium", "24X7Security", 
                        "PowerBackup", "CarParking", "StaffQuarter", "AC", "Wifi", 
                        "Children'splayarea", "LiftAvailable", "BED"]:
                features_dict[col] = categorical_mappings.get(str(data.get(col, "no")).lower(), 0)
            
            elif col in ["Area", "No. of Bedrooms"]:
                try:
                    features_dict[col] = float(data.get(col, 0))
                except ValueError:
                    return jsonify({'error': f'Invalid number format for {col}: {data.get(col)}'})
            else:
                features_dict[col] = 0  # Default for any missing/unexpected features

        # Ensure feature order matches training
        features = np.array([features_dict[col] for col in selected_features]).reshape(1, -1)

        # Apply feature scaling
        features_scaled = scaler.transform(features)

        # Predict price
        prediction = model.predict(features_scaled)[0]

        return jsonify({'predicted_price': f"₹{prediction:,.0f}"})

    except Exception as e:
        logging.error("Error in prediction: %s", str(e))
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
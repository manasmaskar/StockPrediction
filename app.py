# from flask import Flask, request, render_template
# import joblib
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# import logging

# # Initialize the Flask application
# app = Flask(__name__)

# # Load the Gradient Boosting model and the StandardScaler
# try:
#     gb_model = joblib.load('/Users/manasmaskar/Rutgers/Spring24/Utkarsh Project/gradient_boosting_model.pkl')
#     scaler = joblib.load('/Users/manasmaskar/Rutgers/Spring24/Utkarsh Project/standard_scaler.pkl')
# except Exception as e:
#     logging.error(f"Error loading model or scaler: {e}")

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         try:
#             # Receive the form data
#             data = request.form

#             # Extract features from the received data
#             features = [
#                 float(data['open']),
#                 float(data['high']),
#                 float(data['low']),
#                 float(data['volume']),
#                 float(data['macd']),
#                 float(data['signalLine'])
#             ]

#             # Remove feature names from the StandardScaler object
#             scaler_copy = StandardScaler()
#             scaler_copy.mean_ = scaler.mean_
#             scaler_copy.scale_ = scaler.scale_
#             scaler_copy.var_ = scaler.var_
#             scaler_copy.n_features_in_ = scaler.n_features_in_
#             scaler_copy.n_samples_seen_ = scaler.n_samples_seen_
#             scaler_copy.feature_names_in_ = None  # Remove feature names

#             # Use the modified scaler to transform new data
#             scaled_features = scaler_copy.transform([features])

#             # Make predictions using the loaded model
#             prediction = gb_model.predict(scaled_features)[0]

#             # Print debugging information
#             print("Received form data:", data)
#             print("Extracted features:", features)
#             print("Scaled features:", scaled_features)
#             print("Predicted value:", prediction)

#             # Render the result template with the predicted value
#             return render_template('result.html', prediction=prediction)
#         except Exception as e:
#             logging.error(f"Error processing prediction request: {e}")
#             return render_template('error.html', error_message='An unexpected error occurred. Please try again.')
#     else:
#         # Render the index template with the form
#         return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, render_template
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

# Initialize the Flask application
app = Flask(__name__)

# Load the Gradient Boosting model and the StandardScaler
try:
    gb_model = joblib.load('/Users/manasmaskar/Rutgers/Spring24/Utkarsh Project/gradient_boosting_model.pkl')
    scaler = joblib.load('/Users/manasmaskar/Rutgers/Spring24/Utkarsh Project/standard_scaler.pkl')
except Exception as e:
    logging.error(f"Error loading model or scaler: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Receive the form data
            data = request.form

            # Extract features from the received data
            features = [
                float(data['open']),
                float(data['high']),
                float(data['low']),
                float(data['volume']),
                float(data['macd']),
                float(data['signalLine'])
            ]

            # Remove feature names from the StandardScaler object
            scaler_copy = StandardScaler()
            scaler_copy.mean_ = scaler.mean_
            scaler_copy.scale_ = scaler.scale_
            scaler_copy.var_ = scaler.var_
            scaler_copy.n_features_in_ = scaler.n_features_in_
            scaler_copy.n_samples_seen_ = scaler.n_samples_seen_
            scaler_copy.feature_names_in_ = None  # Remove feature names

            # Use the modified scaler to transform new data
            scaled_features = scaler_copy.transform([features])

            # Make predictions using the loaded model
            prediction = gb_model.predict(scaled_features)[0]

            # Render the result template with the predicted value
            return render_template('result.html', prediction=prediction)
        except Exception as e:
            logging.error(f"Error processing prediction request: {e}")
            return render_template('error.html', error_message='An unexpected error occurred. Please try again.')
    else:
        # Render the index template with the form
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


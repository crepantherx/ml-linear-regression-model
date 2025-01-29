from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the saved model


# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to the ML Model Deployment API!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from the POST request
        data = request.get_json()
        if not data or "input" not in data:
            return jsonify({"error": "Please provide input data"}), 400

        # Convert input to a NumPy array
        input_data = np.array(data["input"]).reshape(-1, 1)

        with open("linear_model.pkl", "rb") as f:
            model = pickle.load(f)

        # Make a prediction
        prediction = model.predict(input_data).tolist()

        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
from fastapi import FastAPI
import pickle
import numpy as np



from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

# Load the trained model
with open("linear_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize the FastAPI app
app = FastAPI()



# Define a request model for input validation
class PredictionRequest(BaseModel):
    input: list  # List of numbers (features)


@app.get("/")
def home():
    """Home endpoint."""
    return {"message": "Welcome to the ML Model API!"}


@app.post("/predict")
def predict(request: PredictionRequest):
    """
    Endpoint to make predictions.
    Expects a JSON payload with 'input' as a list of values.
    """
    try:
        # Convert input data to a NumPy array
        input_data = np.array(request.input).reshape(-1, 1)

        # Validate the input
        if input_data.size == 0:
            raise ValueError("Input data is empty.")

        # Perform the prediction
        predictions = model.predict(input_data).tolist()

        return {"predictions": predictions}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


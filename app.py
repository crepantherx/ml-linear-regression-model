from fastapi import FastAPI, Form, HTTPException
from pydantic import BaseModel
import pickle
import uvicorn
import numpy as np
from fastapi.responses import HTMLResponse

app = FastAPI()


class Inputs(BaseModel):
    input: str


@app.get("/", response_class=HTMLResponse)
def read_root():
    return '''
<html>
            <head>
                <title>ML Model Predictor</title>
                <style>
                body {
                    background-color: #fafafa;
                    font-family: Arial, sans-serif;
                    padding: 30px;
                }
                # .container {
                #     display: grid;
                #     place-items: center;
                #     justify-content: center;
                #     align-items: center;
                #     padding: 20px;
                #     border-radius: 8px;
                #     box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
                #     background-color: #ffffff;
                #     max-width: 1000px;
                #     margin: 0 auto;
                # }
                form {
                    display: grid;
                    place-items: center;
                    justify-content: center;
                    align-items: center;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
                    background-color: #ffffff;
                    max-width: 1000px;
                    margin: 0 auto;
                }
                input[type="text"] {
                    margin-top: 20px;
                    padding: 10px;
                    width: 100%;
                    box-sizing: border-box;
                    border-radius: 4px;
                    border: 1px solid #ccc;
                }
                input[type="submit"] {
                    margin-top: 20px;
                    background-color: #007BFF;
                    color: #ffffff;
                    padding: 10px 24px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                }
                </style>
            </head>
            <body>
                <div class="container">
                    
                    <form method="post" action="/predict">
                    <h1>Proof Of Concept</h1>
                     <h3>for machine learning model usability via web</h3>
                        <input name="input" type="text" placeholder="Enter numbers comma-separated">
                        <input type="submit" value="Predict">
                    </form>
                </div>
            </body>
        </html>
    '''


@app.post("/predict", response_class=HTMLResponse)
def predict(input: str = Form(...)):
    try:
        input_list = [float(i) for i in input.split(',')]
        input_data = np.array(input_list).reshape(-1, 1)

        with open("linear_model.pkl", "rb") as f:
            model = pickle.load(f)

        if input_data.size == 0:
            raise ValueError("Input data is empty.")

        predictions = model.predict(input_data).tolist()

        prediction_html = "<h1>Predictions:</h1><br/>" + str(predictions)

        return prediction_html + '''
            <br/>
            <a href="/">Go Back</a>
            <form method="post" action="/predict">
                <input name="input" type="text" placeholder="Enter numbers comma-separated">
                <input type="submit">
            </form>
        '''

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))




if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")

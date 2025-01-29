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
        <form method="post" action="/predict">
            <input name="input" type="text" placeholder="Enter numbers comma-separated">
            <input type="submit">
        </form>
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

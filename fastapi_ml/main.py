# library imports

import uvicorn
from fastapi import FastAPI
from heart import HeartDisease
import numpy as np
import pickle
import pandas as pd

# create the app object
app = FastAPI()
pickle_in = open("../clf.pkl", "rb")
classifier = pickle.load(pickle_in)

# create routes, opens at http://127.0.0.1:8000
@app.get('/')
def index():
    return{'message': 'Hello Archana'}

# route with single parameter http://127.0.0.1:8000/parameter
@app.get('/{name}')
def get_name(name: str):
    return {'message': f"Hello, {name}"}


@app.post('/predict')
def predict_hear_disease(data: HeartDisease):
    data = data.dict()
    age = data['age']
    sex = data['sex']
    cp = data['cp']
    trestbps = data['trestbps']
    chol = data['chol']
    fbs = data['fbs']
    restecg = data['restecg']
    thalach = data['thalach']
    exang = data['exang']
    oldpeak = data['oldpeak']
    slope = data['slope']
    ca = data['ca']
    thal = data['thal']
    # target = data['target']

    prediction = classifier.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    if(prediction[0]>0.5):
        prediction="Heart Disease"
    else:
        prediction="No Heart Disease" 
    return {
        'prediction': prediction
    }       

# run the app with uvicorn server at, http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
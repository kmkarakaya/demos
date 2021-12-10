# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 13:47:05 2020
MURAT KARAKAYA AKADEMİ

NASIL SUNSAM

@author: kmkarakaya
"""
# %%
# Import Modules
from joblib import load
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import numpy as np

from configs import reviewclassification   as rc
#global loaded_end_to_end_model, id_to_category
loaded_end_to_end_model, id_to_category = rc.load_model()




# %% Load Models
def loadIRIS():
    from sklearn.datasets import load_iris
    filename="./models/IRIS.joblib"
    clfUploaded = load(filename)
    dataSet = load_iris()
    labelsNames = list(dataSet.target_names)
    return clfUploaded, labelsNames


# %% Run Server
templates = Jinja2Templates(directory="./html")
app = FastAPI()
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})
# %% SERVE DEMOS
# %% IRIS
@app.get("/iris")
async def read_root(request: Request):
    global clfUploaded, labelsNames 
    clfUploaded, labelsNames = loadIRIS()
    return templates.TemplateResponse("iris.html", {"request": request})

@app.get("/iris/predict")
async def make_prediction(request: Request, L1:float, W1:float,
                          L2:float, W2:float):
    
    testData= np.array([L1,W1,L2,W2]).reshape(-1,4)
    probalities = clfUploaded.predict_proba(testData)[0]
    predicted = np.argmax(probalities)
    probabilty= probalities[predicted]
    predicted = labelsNames [predicted]
    return templates.TemplateResponse("irisprediction.html",
                                      {"request": request,
                                       "probalities": probalities,
                                      "predicted": predicted,
                                      "probabilty": probabilty}  )

# %% Next Demo
# %% Topics
@app.get("/reviewclassification")
async def read_root(request: Request):
    

    raw_data=['Dün aldığım samsung telefon bugün şarj tutmuyor',
            'THY Uçak biletimi değiştirmek için başvurdum.  Kimse geri dönüş yapmadı!']
    predictions=loaded_end_to_end_model.predict(raw_data)
    print(id_to_category[np.argmax(predictions[0])])
    print(id_to_category[np.argmax(predictions[1])])

    return {"text":"Dün aldığım samsung telefon bugün şarj tutmuyor", "class":id_to_category[np.argmax(predictions[0])] }
    #return templates.TemplateResponse("iris.html", {"request": request})





# %% Next Demo






from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
from pydantic import BaseModel
from constant import *
import time
from feature import *
from inference import *


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# class RequestBody(BaseModel):
#     feature:str
#     model:str
#     machineType:str
#     machineId:int


@app.get('/api')
async def ping():
    return {"Hello": "World"}


@app.post('/api/process')
async def process(
    feature : str = Form(...),
    model : str = Form(...),
    machineType : str = Form(...),
    machineId : str = Form(...),
    threshold : str = Form(...),
    files : UploadFile = File(...)
    ):

    content = files.file.read()
    newFilePath = PATH_TMP + str(int(time.time())) + files.filename
    with open(newFilePath, 'wb') as f:
        f.write(content)
    
    # Extract spectrogram
    bytesImage = getImageSpectrogram(convertSpectrogramToVisualizedForm(extractGammatoneFromPath(newFilePath)))

    # Inference
    loss = 0

    if model == ModelType.UNETIDNN.value:
        loss = inferenceUNetIDNN(getFeatureType(feature),getMachineType(machineType),int(machineId), newFilePath)
    elif model == ModelType.AEIDNN.value:
        loss = inferenceIDNN(getFeatureType(feature),getMachineType(machineType),int(machineId), newFilePath)
    # TODO: Other model type

    return{
        'spectrogramImage':bytesImage,
        'loss':loss,
        'threshold':float(threshold)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=int(sys.argv[1]))
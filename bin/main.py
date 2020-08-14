import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disables GPU

import numpy as np
import pandas as pd
import tensorflow as tf

import uvicorn 
from fastapi import FastAPI, Depends
from fastapi.encoders import jsonable_encoder
from keras.models import model_from_json
import base64
import skimage.io
from skimage.color import rgb2gray
from skimage.transform import resize

import structure
from sqlalchemy.orm import Session
from db import SessionLocal, engine
from structure import Records
from pydantic import BaseModel

# loads trained model
with open('../models/model.json', 'r') as json_file:
    cnn_json = json_file.read()
cnn = tf.keras.models.model_from_json(cnn_json)
cnn.load_weights("../models/model.h5")

# decodes base64 image
def decoder(base64_string):
    imgdata = base64.b64decode(base64_string)
    img = skimage.io.imread(imgdata, plugin='imageio')
    return img
  
# calls decoder and transforms to grayscale
def transform_image(input_image):
    decoded = decoder(input_image)
    image_decoded = rgb2gray(decoded)
    
    # resizes
    image_decoded = resize(image_decoded, (28, 28))
    
    # to tensors
    image_decoded = image_decoded.reshape(-1,28,28,1)
    
    return image_decoded
    
# loads base64image
with open("../base64image") as image_file:
    image = image_file.read().replace('\n','')
    
# create database
structure.Base.metadata.create_all(bind=engine)

# start db session
def get_db():
    try: 
        db = SessionLocal()
        yield db
    finally:
        db.close()
app = FastAPI()

# set up API
@app.get("/")
def root():
    return {"MNIST": "Keras OCR"}

@app.get("/predict")
async def predict(db: Session = Depends(get_db)):
    transformed_image = transform_image(image)
    y_pred = cnn.predict_classes(transformed_image)

    rec = Records()
    rec.entries = str(y_pred[0])
    
    db.add(rec)
    db.commit()
    
    return {"the number is": str(y_pred[0]),
            "message": "entry added to database"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000) 

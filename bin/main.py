import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disables GPU

import numpy as np
import pandas as pd
import tensorflow

# API
import uvicorn 
from fastapi import FastAPI, Depends

# images
import base64
import skimage.io
from skimage.color import rgb2gray
from skimage.transform import resize

# database
import structure
from sqlalchemy.orm import Session
from db import SessionLocal, engine
from structure import Records
from pydantic import BaseModel

# loads trained model
from tensorflow.keras.models import model_from_json
with open('../models/model.json', 'r') as json_file:
    cnn_json = json_file.read()
cnn = model_from_json(cnn_json)
cnn.load_weights("../models/model.h5")

# decodes base64 image
def decoder(base64_string):
    imgdata = base64.b64decode(base64_string)
    img = skimage.io.imread(imgdata, plugin='imageio')
    return img
  
# transforms image
def transform_image(input_image):
    # decode
    decoded = decoder(input_image)
    
    #RGB to graycale
    grayscaled = rgb2gray(decoded)
    
    # resizes
    resized = resize(grayscaled, (28, 28))
    
    # to tensors
    tensor_image = resized.reshape(-1,28,28,1)
    
    return tensor_image
    
# loads base64image
with open("../base64image") as image_file:
    image = image_file.read().replace('\n','')
    
# creates database
structure.Base.metadata.create_all(bind=engine)

# starts db session
def get_db():
    try: 
        db = SessionLocal()
        yield db
    finally:
        db.close()
        
# creates FastAPI object
app = FastAPI()

# sets up API
@app.get("/")
def root():
    return {"MNIST": "Keras CNN"}

@app.get("/predict")
async def predict(db: Session = Depends(get_db)):
    # decodes and transforms image
    transformed_image = transform_image(image)
    
    # predicts a number
    y_pred = cnn.predict_classes(transformed_image)
    
    # creates a blueprint of the db structure
    rec = Records()
    
    # stores a string in the 'entries' column
    rec.entries = str(y_pred[0])
    db.add(rec)
    db.commit()
    
    return {"the number is": str(y_pred[0]),
            "message": "entry added to database"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

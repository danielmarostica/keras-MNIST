import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disables GPU
sys.path.insert(1, 'modules/')

# modules
from decode_transform import Transformer
from load_model import load_model

# server-related
import uvicorn
from fastapi import FastAPI, Depends
from pydantic import BaseModel

# database
import db_structure
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError
from db import SessionLocal, engine
from db_structure import Records

# expands BaseModel to validate input
class JSON_image(BaseModel):
    image_str: str

# creates database
def start_db():
    try:
        db_structure.Base.metadata.create_all(bind=engine)
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

@app.post("/predict")
async def predict(image: JSON_image, db: Session = Depends(start_db)):
    # creates image manipulation object
    transformer = Transformer()
    
    # loads predictor
    cnn = load_model('model')

    # decodes and transforms image, which is received through post request and validated with image_str
    image_decoded = transformer.decode(image.image_str)
    image_decoded = transformer.to_tensor(image_decoded)
    
    # predicts a number
    y_pred = cnn.predict_classes(image_decoded)
    
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

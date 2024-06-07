

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import json
from typing import List
from model import predict_disease

import os
import tempfile


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def save_image_to_temp_path(file: UploadFile) -> str:
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    # Create a full path for the new file within the temp directory
    temp_file_path = os.path.join(temp_dir, file.filename)

    # Write the contents of the uploaded file to the temporary file
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file.file.read())

    # Return the full path of the temporary file
    return temp_file_path


@app.get("/ping")
async def ping():
    return "Hello, I am alive"



@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
):
    try:
        image = save_image_to_temp_path(file)

        predicted_class, confidence = predict_disease(image)

        history_entry = {
            'class': predicted_class,
            'confidence': confidence,
        }
        return history_entry
    except Exception as err:
        return {
            "error":str(err)
        }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)


# to run the server: python api/main.py
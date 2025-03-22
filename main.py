from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

import random

import numpy as np
import joblib
from PIL import Image, ImageOps
import io
import os


# Loading the model
try:
    knn_model = joblib.load("./models/knn_model.joblib")
    svm_model = joblib.load("./models/svm_model.joblib")
    print("Models Loaded")
    print(
        f"Number of features expected by knn_model: {knn_model.n_features_in_}")
    print(
        f"Number of features expected by svm_model: {svm_model.n_features_in_}")
except Exception as e:
    print(f"Error loading model: {e}")


app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    print("Static files directory mounted successfully")
except Exception as e:
    print(f"Error mounting static files: {e}")


@app.get("/home")
async def read_index():
    try:
        return FileResponse("static/index.html")
    except Exception as e:
        print(f"Error serving index.html: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-image", response_class=JSONResponse)
async def predict(req: Request):
    """
    Endpoint to predict digit from an uploaded image.
    """
    try:
        data = await req.form()

        file: UploadFile = data['file']
        model_name = data['model']

        # Check content type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, detail="File must be an image.")

        # Read file contents
        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=400, detail="Uploaded file is empty.")

        # Load and process image
        image = Image.open(io.BytesIO(contents)).convert(
            'L')  # Convert to grayscale
        image = ImageOps.invert(image)

        image = image.resize((8, 8), Image.LANCZOS)

        # Convert image to NumPy array and normalize to 0-16
        image_array = np.array(image, dtype=np.float32).reshape(1, -1)

        if model_name == "knn_model":
            prediction = knn_model.predict(image_array)
        else:
            prediction = svm_model.predict(image_array)

        return {"prediction": int(prediction[0])}

    except HTTPException as http_err:
        # Explicitly raised HTTP errors
        raise http_err

    except Exception as e:
        import traceback
        # Log full traceback for internal server errors
        error_details = traceback.format_exc()
        print(f"Error processing image: {str(e)}\n{error_details}")
        raise HTTPException(
            status_code=500, detail="Internal server error while processing image.")

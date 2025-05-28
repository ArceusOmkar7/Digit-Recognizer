import os
import base64
import numpy as np
import joblib
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Define the app
app = FastAPI(title="MNIST Digit Recognizer")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Load model
model_path = os.path.join("models", "random_forest_clf.joblib")
model = joblib.load(model_path)

# Define request model


class ImageData(BaseModel):
    image: str  # base64 encoded image


@app.get("/")
async def read_root():
    return FileResponse("app/static/index.html")


@app.post("/predict")
async def predict(data: ImageData):
    try:
        # Decode base64 image
        image_data = data.image.split(',')[1]
        image_bytes = base64.b64decode(image_data)

        # Open image and convert to grayscale
        image = Image.open(BytesIO(image_bytes)).convert('L')

        # Resize to 28x28 (MNIST format)
        image = image.resize((28, 28))

        # Convert to numpy array
        image_array = np.array(image)

        # Flatten to match MNIST format (1x784)
        flattened = image_array.reshape(1, -1)

        # Make prediction
        prediction = int(model.predict(flattened)[0])

        # Get probabilities for all digits
        probabilities = {}
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(flattened)[0]
                # Create a dictionary with probabilities for each digit
                for digit, prob in enumerate(proba):
                    probabilities[str(digit)] = float(prob)
            except Exception as e:
                print(f"Error getting probabilities: {str(e)}")
                probabilities = {str(i): 0.0 for i in range(10)}
                probabilities[str(prediction)] = 1.0
        else:
            # If model doesn't support probabilities, set 100% for the predicted class
            probabilities = {str(i): 0.0 for i in range(10)}
            probabilities[str(prediction)] = 1.0

        return {
            "prediction": prediction,
            "probabilities": probabilities
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

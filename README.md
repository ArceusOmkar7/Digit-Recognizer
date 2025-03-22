# Digit Recognizer 🤖🖼️

A simple FastAPI project to recognize handwritten digits using KNN and SVM models.

## Installation

1. Clone the repo:
   ```
   git clone https://github.com/your_username/dgr.git
   ```
2. Navigate to the project directory:
   ```
   cd d:\Development\Github\dgr
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the App

Start the FastAPI server:
```
uvicorn main:app --reload
```

Then, open your browser and navigate to:
```
http://127.0.0.1:8000/home
```

## Usage

- Upload an image or draw on the canvas to predict the digit.
- Select your desired model (KNN or SVM) from the dropdown.
- Enjoy! 🎉

## Project Info

- **Backend:** FastAPI, joblib, scikit-learn
- **Frontend:** HTML, Tailwind CSS, Canvas API

Happy coding! 💻😊

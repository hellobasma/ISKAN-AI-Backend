# ISKAN AI Microservice

This repository contains the AI Microservice for the ISKAN real estate platform. It acts as an API Gateway built with FastAPI, connecting the main backend (C# ASP.NET Core) to four distinct AI modules built in Python.

## AI Modules Included
1. **Recommender System (M1)**: Provides property recommendations based on user preferences.
2. **Identity Verification (M2)**: Matches user selfies with National IDs (OCR & Face Matching).
3. **Property Validator (M3)**: Verifies ownership from contracts and detects AI/manipulated property images.
4. **Admin Oversight (M4)**: Generates a combined trust score and dashboard notifications.

## Project Structure
- `main.py`: The FastAPI application and routing endpoints.
- `recommender_Model/`: Logic for M1.
- `ocr_egyptian_ID/`: Logic for M2.
- `Property_Validator_Model/`: Logic for M3.
- `Admin Oversight & Moderation AI Model/`: Logic for M4.
- `requirements.txt`: Python dependencies needed to run the project.

## Setup Instructions

### 1. Install Dependencies
It is highly recommended to use a virtual environment. Run the following command in the terminal to install all required libraries:
```bash
pip install -r requirements.txt
```

### 2. Setup Model Weights
The required YOLO model weights (`.pt` files) are already included in the `ocr_egyptian_ID/models/` directory within this repository. 
Other AI components (like DeepFace and EasyOCR) will automatically download their required system weights from the internet upon the first execution. No manual model downloading is necessary!

### 3. Run the Server
Start the FastAPI server locally by executing:
```bash
python -m uvicorn main:app --reload
```

The server will start at `http://127.0.0.1:8000`. 

### 4. Interactive API Documentation
Once the server is running, you can test the APIs via the interactive Swagger UI at:
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

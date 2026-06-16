import os
import sys
import shutil
import tempfile
import json
import pandas as pd
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.recommender_service import get_similar_properties
from app.ocr_service import verify_user_identity
from app.property_service import PropertyValidator
from app.admin_oversight_service import generate_dashboard_payload  # type: ignore

property_validator_instance = PropertyValidator()

app = FastAPI(title="ISKAN AI Microservice API Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Dummy Database for M1 (To prevent crashes until real DB is linked) ---
mock_data = [
    {"property_id": "1045", "title": "Apartment in Tanta", "location": "Tanta", "property_type": "Apartment", "price_val": 1200000, "price_display": "1,200,000 EGP", "bedrooms": 3, "area_val": 120, "area_display": "120 sqm", "thumbnail_url": "/images/prop_1045.jpg"},
    {"property_id": "890", "title": "Apartment in Tanta", "location": "Tanta", "property_type": "Apartment", "price_val": 1150000, "price_display": "1,150,000 EGP", "bedrooms": 3, "area_val": 115, "area_display": "115 sqm", "thumbnail_url": "/images/prop_890.jpg"}
]
properties_df = pd.DataFrame(mock_data)

# --- Pydantic Models for Data Validation ---
class RecommenderRequest(BaseModel):
    property_id: str

class AdminOversightRequest(BaseModel):
    # Changed to Dict[str, Any] so it accepts the raw complex JSONs directly from M2 and M3
    m2_data: Dict[str, Any]
    m3_data: Dict[str, Any]
    listing_id: str

# --- Pydantic Models for Output Validation (Responses) ---
class RecommenderResponse(BaseModel):
    recommendations: List[Dict[str, Any]]

class IdentityResponse(BaseModel):
    is_match: bool
    national_id: Optional[str]

class PropertyValidationResponse(BaseModel):
    ownership_verified: bool
    image_forensics_flags: Dict[str, Any]

class TrustScore(BaseModel):
    score: int
    label: str

class ImageAnalysis(BaseModel):
    image_index: int
    resolution: str
    authenticity: str
    ai_detection: str

class Notification(BaseModel):
    type: str
    title: str
    time: str

class Report(BaseModel):
    report_title: str
    risk_level: str
    description: str

class AdminOversightResponse(BaseModel):
    listing_id: str
    owner_trust_score: TrustScore
    ai_image_analysis: List[ImageAnalysis]
    dashboard_notifications: List[Notification]
    reports_and_complaints: Report

# --- Helper function ---
def save_upload_file_tmp(upload_file: UploadFile) -> str:
    try:
        suffix = os.path.splitext(upload_file.filename)[1] if upload_file.filename else ""
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, 'wb') as f:
            shutil.copyfileobj(upload_file.file, f)
        return path
    finally:
        upload_file.file.close()

# --- API Endpoints ---

@app.post("/api/recommend", response_model=RecommenderResponse)
async def recommend(request: RecommenderRequest):
    try:
        # Added the missing properties_df argument
        results = get_similar_properties(request.property_id, properties_df=properties_df)
        
        # Handle string JSON just in case it wasn't modified in the AI file
        if isinstance(results, str):
            results = json.loads(results)
            
        if isinstance(results, dict) and "similar_properties" in results:
            return {"recommendations": results["similar_properties"]}
        return {"recommendations": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommender system failed: {str(e)}")

@app.post("/api/verify-identity", response_model=IdentityResponse)
async def verify_identity(
    id_image: UploadFile = File(...),
    selfie_image: UploadFile = File(...)
):
    id_path = None
    selfie_path = None
    try:
        id_path = save_upload_file_tmp(id_image)
        selfie_path = save_upload_file_tmp(selfie_image)
        
        # Added dummy user_form_data to satisfy the function arguments
        result = verify_user_identity(id_path, selfie_path, user_form_data={})
        
        # Mapping the complex AI result to the simple IdentityResponse Pydantic model
        mapped_response = {
            "is_match": result.get("face_verification", {}).get("is_match", False),
            "national_id": result.get("extracted_identity", {}).get("national_id", None)
        }
        return mapped_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Identity verification failed: {str(e)}")
    finally:
        if id_path and os.path.exists(id_path):
            os.remove(id_path)
        if selfie_path and os.path.exists(selfie_path):
            os.remove(selfie_path)

@app.post("/api/validate-property", response_model=PropertyValidationResponse)
async def validate_property(
    contract_file: UploadFile = File(...),
    property_images: List[UploadFile] = File(...),
    verified_owner_national_id: str = Form(...)
):
    contract_path = None
    image_paths = []
    try:
        contract_path = save_upload_file_tmp(contract_file)
        for img in property_images:
            image_paths.append(save_upload_file_tmp(img))
            
        result = property_validator_instance.validate_property_listing(contract_path, image_paths, verified_owner_national_id)
        
        # Mapping the complex AI result to the simple Pydantic model
        mapped_response = {
            "ownership_verified": result.get("document_analysis", {}).get("ownership_verified", False),
            "image_forensics_flags": {"flags": result.get("image_forensics", {}).get("flags", [])}
        }
        return mapped_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Property validation failed: {str(e)}")
    finally:
        if contract_path and os.path.exists(contract_path):
            os.remove(contract_path)
        for path in image_paths:
            if path and os.path.exists(path):
                os.remove(path)

@app.post("/api/admin-oversight", response_model=AdminOversightResponse)
async def admin_oversight(request: AdminOversightRequest):
    try:
        # Pydantic dict structure directly passed so logic can search inside it
        result = generate_dashboard_payload(
            request.m2_data, 
            request.m3_data, 
            request.listing_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Admin oversight failed: {str(e)}")
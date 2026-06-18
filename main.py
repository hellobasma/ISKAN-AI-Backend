import os
import sys
import shutil
import tempfile
import json
import pandas as pd
import requests
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field # تم إضافة Field هنا

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

# --- Real Database Connection (Fetching from C# API) ---
def load_properties_from_backend():
    backend_url = "https://isskan-1.runasp.net/api/Property/GetAll?PageIndex=1&PageSize=1000" 
    
    try:
        response = requests.get(backend_url)
        
        if response.status_code == 200:
            json_response = response.json()
            properties_list = json_response.get("data", [])
            df = pd.DataFrame(properties_list)
            
            if not df.empty:
                # 1. تغيير الأسماء (قاموس الترجمة الشامل)
                df = df.rename(columns={
                    "id": "property_id",            
                    "pricePerMonth": "price_val",   
                    "propertyType": "property_type", 
                    "address": "location",           
                    "bedroomsNumber": "bedrooms",   
                    "bathroomsNumber": "bathrooms", 
                    "roomsNumber": "rooms",
                    "mainImageUrl": "thumbnail_url"  # 👈 اللمسة الأخيرة للصورة
                })
                
                # 2. قيم افتراضية للحاجات الناقصة عشان الموديل ميزعلش
                if 'area_val' not in df.columns:
                    df['area_val'] = 100 
                if 'thumbnail_url' not in df.columns:
                    df['thumbnail_url'] = ""
                    
                # 🧹 تنظيف الداتا: استخراج اسم المدينة من العنوان عشان الموديل يفهم
                if 'location' in df.columns:
                    df['location'] = df['location'].astype(str)
                    # السطر اللي بيستخرج المنطقة
                    df['location_clean'] = df['location'].apply(lambda x: " ".join(x.split(',')[-2:]).strip() if ',' in x else x)
    
                    # 💡 حطي السطر ده هنا فوراً عشان يضمن إن مفيش قيم فاضية
                    df['location_clean'] = df['location_clean'].replace('', 'Unknown')

                if 'property_type' in df.columns:
                    # تحويل الأصفار لكلمة Room عشان تتوحد مع الداتا
                    df['property_type'] = df['property_type'].astype(str).replace('0', 'Room')

                # 4. تنظيف البيانات الرقمية
                numeric_cols = ['price_val', 'bedrooms', 'bathrooms', 'rooms']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
                # 🚫 فلترة حاسمة: طرد أي عقار سعره معدي الـ 50 ألف عشان ميبوظش الموديل
                df = df[df['price_val'] < 50000]
                    
                # 5. تجهيز العرض النهائي للـ JSON
                if 'price_val' in df.columns:
                    df['price_display'] = df['price_val'].astype(int).astype(str)
                if 'area_val' in df.columns:
                    df['area_display'] = df['area_val'].astype(int).astype(str) + " m²"

            print(f"✅ Successfully loaded {len(df)} properties for recommendation.")
            return df
        else:
            print(f"❌ Failed to fetch data. Status Code: {response.status_code}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Error connecting to backend API: {e}")
        return pd.DataFrame()

properties_df = load_properties_from_backend()

# --- Pydantic Models for Data Validation ---
class RecommenderRequest(BaseModel):
    property_id: str


class IdentityVerifyRequest(BaseModel):
    id_image_url: str
    selfie_image_url: str


# 1. حل مشكلة الربط مع C# باستخدام Alias
class AdminOversightRequest(BaseModel):
    m2_data: Dict[str, Any] = Field(..., alias="M2Data")
    m3_data: Dict[str, Any] = Field(..., alias="M3Data")
    listing_id: str = Field(..., alias="ListingId")

    class Config:
        populate_by_name = True # يسمح للـ API بقراءة الطريقتين

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
        results = get_similar_properties(request.property_id, properties_df=properties_df)
        
        if isinstance(results, str):
            results = json.loads(results)
            
        # --- التعديل الذكي لفك التغليف ---
        if isinstance(results, dict):
            if "similar_properties" in results:
                return {"recommendations": results["similar_properties"]}
            elif "recommendations" in results:
                return {"recommendations": results["recommendations"]}
                
        # لو الداتا راجعة كلستة مباشرة
        return {"recommendations": results}
        
    except Exception as e:
        import traceback
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=f"Recommender system failed: {str(e)}")

@app.post("/api/verify-identity", response_model=IdentityResponse)
async def verify_identity(request: IdentityVerifyRequest):
    try:
        # الموديل بتاعك هيستقبل الروابط مباشرة
        result = verify_user_identity(
            id_path=request.id_image_url, 
            selfie_path=request.selfie_image_url, 
            user_form_data={}
        )
        
        national_id = result.get("extracted_identity", {}).get("national_id", None)
        
        if not national_id or str(national_id).strip() == "":
            raise HTTPException(status_code=400, detail="Could not extract ID from provided URL.")
            
        return {
            "is_match": result.get("face_verification", {}).get("is_match", False),
            "national_id": national_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Identity verification failed: {str(e)}")

@app.post("/api/validate-property", response_model=PropertyValidationResponse)
async def validate_property(
    contract_file: UploadFile = File(...),
    property_images: List[UploadFile] = File(...),
    verified_owner_national_id: Optional[str] = Form(None) # السماح بكونه اختياري مبدئياً للتحكم في الإيرور
):
    # 2. حماية ذكية ضد الـ Error 500
    if not verified_owner_national_id or verified_owner_national_id.strip() == "":
        raise HTTPException(status_code=400, detail="Owner national ID is required for AI property validation.")

    contract_path = None
    image_paths = []
    try:
        contract_path = save_upload_file_tmp(contract_file)
        for img in property_images:
            image_paths.append(save_upload_file_tmp(img))
            
        result = property_validator_instance.validate_property_listing(contract_path, image_paths, verified_owner_national_id)
        
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
        # استدعاء الدالة الأصلية
        result = generate_dashboard_payload(
            request.m2_data, 
            request.m3_data, 
            request.listing_id
        )
        
        # 3. التدخل السريع لحساب Trust Score حقيقي بناءً على الداتا
        calculated_score = 100
        
        # الخصم على عدم وجود صور
        image_count = request.m2_data.get("image_count", 0)
        if image_count == 0:
            calculated_score -= 40
        elif image_count < 3:
            calculated_score -= 15
            
        # الخصم على عدم التوثيق
        ver_status = request.m3_data.get("verification_status", "Unverified")
        if ver_status != "Approved" and ver_status != "Verified":
            calculated_score -= 30
            
        # ضمان إن السكور ميبقاش تحت الصفر
        calculated_score = max(0, calculated_score)
        
        # تحديد التقييم
        if calculated_score >= 80:
            label = "Excellent"
        elif calculated_score >= 50:
            label = "Average"
        else:
            label = "Poor"
            
        # تحديث النتيجة بالسكور الحقيقي
        result["owner_trust_score"] = {"score": calculated_score, "label": label}
        
        # تحديث الإشعارات لو العقار مشبوه
        if calculated_score < 50:
            result["dashboard_notifications"] = [
                {"type": "Warning", "title": f"Listing {request.listing_id} needs manual review", "time": "Just now"}
            ]
            result["reports_and_complaints"] = {
                "report_title": "Suspicious Listing Detected",
                "risk_level": "High Priority",
                "description": "This listing lacks sufficient images or is unverified."
            }
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Admin oversight failed: {str(e)}")
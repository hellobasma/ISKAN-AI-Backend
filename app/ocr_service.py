"""
Identity Verification Pipeline
------------------------------
This module handles the integration between Face Matching and Identity Extraction
for the ISKAN real estate platform.

It uses DeepFace for robust face matching, specifically configured to handle
complex backgrounds and low-resolution/grayscale photos common in Egyptian National IDs.
"""

import logging
import requests
import tempfile
import os
from deepface import DeepFace
from app.ocr_utils import detect_and_process_id_card

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_image_temp(url: str) -> str:
    """
    Downloads an image from a URL and saves it to a temporary file.
    Returns the path to the temporary file.
    """
    try:
        logger.info(f"Downloading image from URL: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise exception for bad status codes
        
        # Create a temporary file
        fd, temp_path = tempfile.mkstemp(suffix=".jpg")
        with os.fdopen(fd, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return temp_path
    except Exception as e:
        logger.error(f"Failed to download image from {url}: {e}")
        raise ValueError(f"Could not download image from provided URL. Check if the URL is accessible.")

def identity_extraction(id_image_path: str) -> dict:
    """
    Wrapper for the existing identity extraction module.
    Calls the YOLO and EasyOCR pipeline from utils.py to extract all information
    from the Egyptian ID card.
    """
    logger.info("Running existing OCR logic on the ID image...")
    try:
        first_name, second_name, full_name, national_id, address, birth, gov, gender, photo_path = detect_and_process_id_card(id_image_path)
        return {
            "status": "Data Validated",
            "firstName": first_name,
            "lastName": second_name,
            "fullName": full_name,
            "national_id": national_id,
            "address": address,
            "dob": birth,
            "governorate": gov,
            "gender": gender,
            "photo_path": photo_path
        }
    except Exception as e:
        logger.error(f"OCR Extraction failed: {e}", exc_info=True)
        return {
            "status": "Extraction Failed",
            "error": str(e)
        }

def calculate_similarity_percentage(distance: float, distance_metric: str = "cosine") -> float:
    """
    Converts a distance metric into a similarity percentage.
    """
    if distance_metric == "cosine":
        similarity = (1.0 - distance) * 100.0
        return max(0.0, min(100.0, similarity))
    else:
        logger.warning(f"Similarity calculation for metric '{distance_metric}' is not explicitly defined. Using raw distance fallback.")
        return max(0.0, (1.0 - distance) * 100.0)

def verify_user_identity(id_path: str, selfie_path: str, user_form_data: dict) -> dict:
    """
    Master controller for the Identity Verification Pipeline.
    Now supports both local file paths and URLs.
    """
    logger.info("Starting identity verification process...")
    
    response = {
        "transaction_status": "Failed",
        "face_verification": {
            "is_match": False,
            "similarity_score": "0.00%",
            "faces_detected": False
        },
        "extracted_identity": {
            "status": "Pending",
            "national_id": None
        },
        "system_message": "Initialization."
    }

    local_id_path = None
    local_selfie_path = None

    try:
        # 1. 🌐 Download images if they are URLs
        if id_path.startswith("http://") or id_path.startswith("https://"):
            local_id_path = download_image_temp(id_path)
        else:
            local_id_path = id_path

        if selfie_path.startswith("http://") or selfie_path.startswith("https://"):
            local_selfie_path = download_image_temp(selfie_path)
        else:
            local_selfie_path = selfie_path

        # Step A: Run Identity Extraction
        logger.info("Running Identity Extraction first to get YOLO ID photo crop...")
        extracted_data = identity_extraction(local_id_path)
        
        response["extracted_identity"] = {k: v for k, v in extracted_data.items() if k != "photo_path"}
        extracted_photo_path = extracted_data.get("photo_path")
        
        if not extracted_photo_path:
            response["transaction_status"] = "Failed"
            response["system_message"] = "OCR process failed to extract a photo box from the ID card."
            logger.warning("No photo_path returned from identity_extraction.")
            return response

        # Step B: Run Face Matching
        model_name = "Facenet512" 
        detector_backend = "retinaface"
        distance_metric = "cosine"
        
        logger.info(f"Running DeepFace verification on extracted photo. Model: {model_name}, Detector: {detector_backend}")
        
        result = DeepFace.verify(
            img1_path=extracted_photo_path,
            img2_path=local_selfie_path,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            enforce_detection=True
        )

        response["face_verification"]["faces_detected"] = True
        distance = result.get("distance", 1.0)
        similarity_percentage = calculate_similarity_percentage(distance, distance_metric)
        response["face_verification"]["similarity_score"] = f"{similarity_percentage:.2f}%"

        match_threshold = 55.0
        
        if similarity_percentage >= match_threshold:
            response["face_verification"]["is_match"] = True
            response["transaction_status"] = "Success"
            response["system_message"] = "Face match successful and identity data extracted."
        else:
            response["face_verification"]["is_match"] = False
            response["transaction_status"] = "Failed"
            response["system_message"] = f"Face match failed. Similarity ({similarity_percentage:.2f}%) is below the required threshold ({match_threshold}%)."

    except ValueError as ve:
        logger.error(f"Detection or Download error: {ve}")
        response["face_verification"]["faces_detected"] = False
        response["transaction_status"] = "Failed"
        response["system_message"] = str(ve)
        
    except Exception as e:
        logger.error(f"Unexpected error during face verification: {e}", exc_info=True)
        response["transaction_status"] = "Failed"
        response["system_message"] = "An unexpected server error occurred during face verification. Please try again."
        
    finally:
        # 🧹 Cleanup temporary files ONLY if they were downloaded from URLs
        if local_id_path and id_path != local_id_path and os.path.exists(local_id_path):
            os.remove(local_id_path)
            logger.info("Cleaned up temp ID file.")
        if local_selfie_path and selfie_path != local_selfie_path and os.path.exists(local_selfie_path):
            os.remove(local_selfie_path)
            logger.info("Cleaned up temp Selfie file.")

    return response
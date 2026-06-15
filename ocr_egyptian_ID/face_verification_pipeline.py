"""
Identity Verification Pipeline
------------------------------
This module handles the integration between Face Matching and Identity Extraction
for the ISKAN real estate platform.

It uses DeepFace for robust face matching, specifically configured to handle
complex backgrounds and low-resolution/grayscale photos common in Egyptian National IDs.
"""

import logging
from deepface import DeepFace
from utils import detect_and_process_id_card
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    For 'cosine' distance, a distance of 0 means perfectly identical faces,
    and a distance of 1 means completely orthogonal (different) faces.
    
    Formula: Similarity = (1.0 - distance) * 100
    """
    if distance_metric == "cosine":
        similarity = (1.0 - distance) * 100.0
        return max(0.0, min(100.0, similarity))
    else:
        # If other metrics are used (like euclidean), this calculation would need to be adjusted.
        logger.warning(f"Similarity calculation for metric '{distance_metric}' is not explicitly defined. Using raw distance fallback.")
        return max(0.0, (1.0 - distance) * 100.0)

def verify_user_identity(id_image_path: str, selfie_image_path: str, user_form_data: dict) -> dict:
    """
    Master controller for the Identity Verification Pipeline.
    
    Steps:
    A: Run the Face Matching module (DeepFace).
    B: If Face Match fails (similarity < threshold), halt the process.
    C: If Face Match passes, call the existing identity_extraction function.
    
    Args:
        id_image_path (str): File path to the National ID image.
        selfie_image_path (str): File path to the User Selfie image.
        user_form_data (dict): Dictionary containing user details from the UI form.
        
    Returns:
        dict: A comprehensive JSON-serializable response.
    """
    logger.info("Starting identity verification process...")
    
    # Initialize the default unified response structure
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

    try:
        # Step A: Call existing identity_extraction module FIRST to get the YOLO cropped photo
        logger.info("Running Identity Extraction first to get YOLO ID photo crop...")
        extracted_data = identity_extraction(id_image_path)
        
        # Save the OCR data but exclude the internal photo_path from the final JSON
        response["extracted_identity"] = {k: v for k, v in extracted_data.items() if k != "photo_path"}
        
        extracted_photo_path = extracted_data.get("photo_path")
        
        if not extracted_photo_path:
            response["transaction_status"] = "Failed"
            response["system_message"] = "OCR process failed to extract a photo box from the ID card."
            logger.warning("No photo_path returned from identity_extraction.")
            return response

        # Step B: Run the Face Matching module using the EXTRACTED ID PHOTO
        model_name = "Facenet512" 
        detector_backend = "retinaface"
        distance_metric = "cosine"
        
        logger.info(f"Running DeepFace verification on extracted photo. Model: {model_name}, Detector: {detector_backend}")
        
        result = DeepFace.verify(
            img1_path=extracted_photo_path,
            img2_path=selfie_image_path,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            enforce_detection=True # Ensures exception is thrown if face is not found
        )

        response["face_verification"]["faces_detected"] = True

        # Extract the calculated distance and convert to a similarity percentage
        distance = result.get("distance", 1.0)
        similarity_percentage = calculate_similarity_percentage(distance, distance_metric)
        response["face_verification"]["similarity_score"] = f"{similarity_percentage:.2f}%"

        logger.info(f"Face distance: {distance}, Similarity Percentage: {similarity_percentage:.2f}%")

        # Custom Threshold Adjustment (55-60% as requested)
        match_threshold = 55.0
        
        if similarity_percentage >= match_threshold:
            # Face Match Passes
            response["face_verification"]["is_match"] = True
            response["transaction_status"] = "Success"
            response["system_message"] = "Face match successful and identity data extracted."
            logger.info("Face match passed.")
        else:
            # Face Match Fails - halt process
            response["face_verification"]["is_match"] = False
            response["transaction_status"] = "Failed"
            response["system_message"] = f"Face match failed. Similarity ({similarity_percentage:.2f}%) is below the required threshold ({match_threshold}%)."
            logger.warning("Face match failed due to low similarity.")

    except ValueError as ve:
        # DeepFace raises a ValueError when a face is not detected in one of the images
        logger.error(f"Face detection error: {ve}")
        response["face_verification"]["faces_detected"] = False
        response["transaction_status"] = "Failed"
        response["system_message"] = "Face detection failed. Ensure both the ID and selfie clearly show a face, and that the front of the ID was uploaded."
        
    except Exception as e:
        # Broad exception catch to prevent server crashes on unexpected DeepFace/system errors
        logger.error(f"Unexpected error during face verification: {e}", exc_info=True)
        response["transaction_status"] = "Failed"
        response["system_message"] = "An unexpected server error occurred during face verification. Please try again."

    return response

if __name__ == "__main__":
    # Example Usage (for testing purposes)
    # import json
    # sample_id_path = "sample_id_front.jpg"
    # sample_selfie_path = "sample_selfie.jpg"
    # form_data = {"user_id": "12345", "name": "Ahmed"}
    # result = verify_user_identity(sample_id_path, sample_selfie_path, form_data)
    # print(json.dumps(result, indent=2))
    pass

import os
import re
import cv2
import json
import easyocr
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import pypdfium2 as pdfium

class PropertyValidator:
    def __init__(self):
        """
        Initializes the Property Validator Model.
        Sets up the OCR engine and the list of suspicious software keywords.
        """
        # Initialize EasyOCR reader for Arabic and English (for numbers)
        # Using gpu=False to make it accessible without CUDA, but can be configured
        print("Initializing OCR Engine (this might take a moment if it's the first run)...")
        self.reader = easyocr.Reader(['ar', 'en'], gpu=False)
        
        # Known software keywords that indicate manipulation or AI generation
        self.suspicious_software_keywords = [
            'photoshop', 'lightroom', 'illustrator', 'after effects',
            'midjourney', 'dall-e', 'stable diffusion', 'canva', 'gimp',
            'snapseed', 'vsco', 'fotor', 'picsart', 'ai', 'generative'
        ]

    def _convert_pdf_to_images(self, pdf_path):
        """Converts a PDF file into a list of PIL Images using pypdfium2."""
        images = []
        try:
            pdf = pdfium.PdfDocument(pdf_path)
            for i in range(len(pdf)):
                page = pdf[i]
                # Scale up the image resolution (2x) for better OCR performance
                bitmap = page.render(scale=2)
                img = bitmap.to_pil()
                images.append(img)
                page.close()
            pdf.close()
        except Exception as e:
            print(f"Error converting PDF to images: {e}")
        return images

    def _preprocess_image_for_ocr(self, image):
        """Preprocesses an image using OpenCV for better OCR results."""
        # Convert PIL image to OpenCV format (numpy array)
        if isinstance(image, Image.Image):
            img_np = np.array(image)
            # PIL is RGB, OpenCV is BGR
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            # Assuming it's a file path
            img_bgr = cv2.imread(image)
            if img_bgr is None:
                return None
                
        # Convert to grayscale
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Apply Adaptive Thresholding
        # Helps deal with varying lighting and shadows often found in scanned documents
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return thresh

    def verify_title_deed(self, contract_path, verified_owner_national_id):
        """
        Module 1: Title Deed Verification
        Extracts text and searches for the verified national ID.
        """
        images_to_process = []
        
        if contract_path.lower().endswith('.pdf'):
            images_to_process = self._convert_pdf_to_images(contract_path)
        else:
            try:
                img = Image.open(contract_path)
                images_to_process.append(img)
            except Exception as e:
                return {
                    "ownership_verified": False,
                    "extracted_ids_found": [],
                    "message": f"Failed to load contract document: {str(e)}"
                }

        extracted_text = ""
        for img in images_to_process:
            preprocessed = self._preprocess_image_for_ocr(img)
            if preprocessed is not None:
                # Run EasyOCR
                # detail=0 returns only the text strings, not bounding boxes
                results = self.reader.readtext(preprocessed, detail=0)
                extracted_text += " ".join(results) + " "

        # Regex to find exactly 14-digit sequences
        # \b ensures we match exactly 14 digits as a whole word, without surrounding letters/numbers
        id_pattern = re.compile(r'\b\d{14}\b')
        found_ids = id_pattern.findall(extracted_text)
        
        # Remove duplicates while preserving list format
        found_ids = list(set(found_ids))
        
        is_verified = verified_owner_national_id in found_ids
        
        if is_verified:
            message = "Owner ID matched with the contract successfully."
        elif found_ids:
            message = "Document processed, but verified Owner ID was not found among the extracted IDs."
        else:
            message = "No 14-digit IDs could be found in the document."

        return {
            "ownership_verified": is_verified,
            "extracted_ids_found": found_ids,
            "message": message
        }

    def analyze_property_images(self, property_images_paths_list):
        """
        Module 2: Property Image Forensics
        Analyzes EXIF data of property images to detect screenshots or manipulation.
        """
        total_images = len(property_images_paths_list)
        authentic_images = 0
        suspicious_images = 0
        flags = []

        for index, image_path in enumerate(property_images_paths_list):
            try:
                img = Image.open(image_path)
                
                # Extract EXIF metadata
                exif_data = img.getexif()
                
                # Check alternative method if .getexif() fails
                if not exif_data and hasattr(img, '_getexif') and img._getexif() is not None:
                    exif_data = img._getexif()

                if not exif_data:
                    suspicious_images += 1
                    flags.append({
                        "image_index": index,
                        # "image_path": image_path, # Optional: Include path
                        "reason": "Missing EXIF data - Potential screenshot from the internet or stripped metadata."
                    })
                    continue

                # Parse EXIF tags into human-readable dictionary
                parsed_exif = {}
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    parsed_exif[tag_name] = value

                # Check for Software/Processing tags indicating manipulation
                software_tag = str(parsed_exif.get('Software', '')).lower()
                
                is_manipulated = False
                for keyword in self.suspicious_software_keywords:
                    if keyword in software_tag:
                        is_manipulated = True
                        suspicious_images += 1
                        flags.append({
                            "image_index": index,
                            "reason": f"Manipulated/AI-Generated - Detected editing software signature: {software_tag}"
                        })
                        break
                
                if not is_manipulated:
                    authentic_images += 1

            except Exception as e:
                suspicious_images += 1
                flags.append({
                    "image_index": index,
                    "reason": f"Failed to process image file. Error: {str(e)}"
                })

        return {
            "total_images_analyzed": total_images,
            "authentic_images": authentic_images,
            "suspicious_images": suspicious_images,
            "flags": flags
        }

    def validate_property_listing(self, contract_image_path, property_images_paths_list, verified_owner_national_id):
        """
        Module 3: Integration & Main Controller
        Runs both modules sequentially and aggregates the results into a JSON object.
        """
        # Run Module 1: Document OCR Analysis
        document_analysis = self.verify_title_deed(contract_image_path, verified_owner_national_id)
        
        # Run Module 2: Image Forensics
        image_forensics = self.analyze_property_images(property_images_paths_list)
        
        # Determine overall validation status
        # Logic: Ownership must be verified AND there shouldn't be suspicious images
        if document_analysis["ownership_verified"] and image_forensics["suspicious_images"] == 0:
            status = "Approved"
        else:
            status = "Pending_Manual_Review"

        response = {
            "listing_validation_status": status,
            "document_analysis": document_analysis,
            "image_forensics": image_forensics
        }

        return response

if __name__ == "__main__":
    # -------------------------------------------------------------
    # Example Usage for Backend Integration
    # -------------------------------------------------------------
    
    # 1. Initialize the validator (Should be initialized once in the backend lifecycle)
    validator = PropertyValidator()
    
    # 2. Setup Dummy Payload
    sample_contract_path = "sample_contract.pdf"
    sample_property_images = ["room1.jpg", "kitchen.png", "exterior.jpg"]
    sample_owner_id = "12345678901234"
    
    print("\n--- Example Model Output ---")
    
    # Simulate the function call (Commented out because files don't exist yet)
    '''
    result_json = validator.validate_property_listing(
        contract_image_path=sample_contract_path,
        property_images_paths_list=sample_property_images,
        verified_owner_national_id=sample_owner_id
    )
    
    # The output will be neatly formatted JSON as requested
    print(json.dumps(result_json, indent=2, ensure_ascii=False))
    '''
    print("Script ran successfully. Ready for integration.")

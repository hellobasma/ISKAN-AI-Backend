import sys
import json
from face_verification_pipeline import verify_user_identity

def main():
    if len(sys.argv) < 3:
        print("Usage: python test_pipeline.py <path_to_id_image> <path_to_selfie_image>")
        print("\nExample:")
        print("python test_pipeline.py id_card.jpg selfie.jpg")
        sys.exit(1)
        
    id_image_path = sys.argv[1]
    selfie_image_path = sys.argv[2]
    
    # Dummy form data as this is just a test for face verification
    form_data = {
        "user_id": "test_user_001",
        "name": "Test User"
    }
    
    print(f"Testing Face Verification Pipeline...")
    print(f"ID Image: {id_image_path}")
    print(f"Selfie Image: {selfie_image_path}")
    print("-" * 50)
    
    # Run the pipeline
    result = verify_user_identity(id_image_path, selfie_image_path, form_data)
    
    print("\n--- Final Pipeline Result ---")
    print(json.dumps(result, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()

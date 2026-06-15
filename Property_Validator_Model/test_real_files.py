import json
from property_validator import PropertyValidator

def test_real_files():
    print("Initializing Validator...")
    validator = PropertyValidator()

    contract_path = r"C:\Users\Basma\Downloads\Property_Validator_Model\عقد بيع شقة سكنية ابتدائي - ملف اختبار.pdf"
    images_paths = [r"C:\Users\Basma\Downloads\Property_Validator_Model\property_image_test.png"]
    
    # Since we don't know the exact ID in the document yet, we'll use a placeholder.
    # The output will tell us what 14-digit IDs it found.
    dummy_owner_id = "00000000000000"

    print("\nRunning Validation on real files...")
    result = validator.validate_property_listing(contract_path, images_paths, dummy_owner_id)
    
    print("\n--- Validation Result ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    test_real_files()

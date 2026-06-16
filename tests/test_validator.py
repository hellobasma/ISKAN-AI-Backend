import os
import cv2
import numpy as np
from PIL import Image, ExifTags
import json
from property_validator import PropertyValidator

def create_dummy_files():
    # 1. Create a dummy contract image with a 14-digit ID
    # White background
    img = np.ones((500, 800, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Property Contract', (50, 100), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, 'Owner ID: 12345678901234', (50, 200), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite('dummy_contract.jpg', img)

    # 2. Create an authentic property image
    img_authentic = Image.new('RGB', (100, 100), color = 'red')
    # Adding basic EXIF data using PIL is a bit tricky, but we can at least save it.
    # To simulate an authentic image, it just shouldn't have 'Software' tags with bad words, 
    # but it needs SOME exif data to pass the "Missing EXIF" check.
    exif = img_authentic.getexif()
    exif[0x010F] = "Camera Manufacturer" # Make
    img_authentic.save('dummy_authentic.jpg', exif=exif)

    # 3. Create a manipulated property image
    img_manipulated = Image.new('RGB', (100, 100), color = 'blue')
    exif_manip = img_manipulated.getexif()
    # Software tag is 0x0131
    exif_manip[0x0131] = "Adobe Photoshop CC"
    img_manipulated.save('dummy_manipulated.jpg', exif=exif_manip)

    # 4. Create a screenshot (no exif)
    img_screenshot = Image.new('RGB', (100, 100), color = 'green')
    img_screenshot.save('dummy_screenshot.png')

def test():
    print("Creating dummy test files...")
    create_dummy_files()

    print("Initializing Validator...")
    validator = PropertyValidator()

    contract_path = 'dummy_contract.jpg'
    images_paths = ['dummy_authentic.jpg', 'dummy_manipulated.jpg', 'dummy_screenshot.png']
    owner_id = '12345678901234'

    print("\nRunning Validation...")
    result = validator.validate_property_listing(contract_path, images_paths, owner_id)
    
    print("\nValidation Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Clean up
    for f in [contract_path] + images_paths:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    test()

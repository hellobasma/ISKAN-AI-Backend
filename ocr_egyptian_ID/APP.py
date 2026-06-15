import os
import tempfile
from PIL import Image
import streamlit as st
from utils import detect_and_process_id_card

# Streamlit configuration
st.set_page_config(page_title='ID Egyptian Card ', page_icon='💳', layout='wide')

# Initialize session state for navigation
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Home"

# Sidebar navigation menu
tabs = ["Home", "Guide"]
selected_tab = st.sidebar.radio("Navigation", tabs)

# Update the session state with the selected tab
st.session_state.current_tab = selected_tab

# Home Tab
if st.session_state.current_tab == "Home":
    uploaded_file = st.sidebar.file_uploader("Upload an ID card image",
                                             type=['webp', 'jpg', 'tif', 'tiff', 'png', 'mpo', 'bmp', 'jpeg', 'dng', 'pfm'])

    # If no file is uploaded, display the HOME image
    if not uploaded_file:
        st.image("ocr2.png", use_container_width=True)
    else:
        # If a file is uploaded, process it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        image = Image.open(temp_file_path)

        st.subheader('Egyptian ID Card EXTRACTING, OCR 💳')
        st.sidebar.image(image)

        try:
            # Call the detect_and_process_id_card function
            first_name, second_name, Full_name, national_id, address, birth, gov, gender, photo_path = detect_and_process_id_card(temp_file_path)
            st.image(Image.open("d2.jpg"), use_container_width=True)
            st.markdown("---")
            st.markdown(" ## WORDS EXTRACTED : ")
            st.write(f"First Name: {first_name}")
            st.write(f"Second Name: {second_name}")
            st.write(f"Full Name: {Full_name}")
            st.write(f"National ID: {national_id}")
            st.write(f"Address: {address}")
            st.write(f"Birth Date: {birth}")
            st.write(f"Governorate: {gov}")
            st.write(f"Gender: {gender}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            os.remove(temp_file_path)

# Documentation Tab
elif st.session_state.current_tab == "Guide":
    st.title("How to use our application 📖")
    st.write("""
    ## Project Overview:
    This application processes Egyptian ID cards to extract key information, including names, addresses, and national IDs.  
    It also decodes the national ID to provide additional details like birth date, governorate, and gender.

    ## Features:
    - **ID Card Detection**: Automatically detects and crops the ID card from the image.
    - **Field Detection**: Identifies key fields such as first name, last name, address, and serial number.
    - **Text Extraction**: Extracts Arabic and English text using EasyOCR.
    - **National ID Decoding**: Decodes the ID to extract:
        - Birth Date
        - Governorate
        - Gender
        - Birthplace
        - Location
        - Nationality

    ## How It Works:
    1. **Upload an Image**: Upload an image of the ID card using the sidebar.
    2. **Detection and Extraction**:
        - YOLO models detect the ID card and its fields.
        - EasyOCR extracts text from the identified fields.
    3. **Result Presentation**:
        - Outputs extracted information such as full name, address, and national ID details.
    4. **ID Decoding**:
        - Decodes the national ID to reveal demographic details.

    ## Steps to Use:
    - Get your image ready.
    - Click on Home.
    - Upload an Egyptian ID card image.
    - View the extracted information and analysis.
        
    ## هI HOPE YOU ENJOY THE EXPERIENCE 💖
    """)

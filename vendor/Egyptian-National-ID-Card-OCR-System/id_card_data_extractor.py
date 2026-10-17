"""
Egyptian National ID Card OCR Data Extraction Module

This module provides functionality to extract and process text data from both sides
of Egyptian national ID cards using OCR (Optical Character Recognition).

Features:
- Front side processing: Name, address, ID number, birth date, governorate
- Back side processing: Job, workplace, gender, religion, marital status, spouse name
- Arabic text recognition with specialized handling for Egyptian ID formats
- Data validation and normalization

Dependencies:
- OpenCV for image processing
- PIL for image manipulation
- pytesseract for OCR
- Custom modules: pob (place of birth), gender (gender detection)
"""

from PIL import Image
import cv2
import numpy as np
import os
from pytesseract import image_to_string
import pytesseract

# Import custom modules
import pob
import gender as gender_detector

# # Configure Tesseract OCR paths (should be moved to config file)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

# Global variables to store extracted ID card data
# TODO: Consider using a class-based approach instead of global variables
extracted_photo = np.zeros((300, 225))  # Extracted ID photo
extracted_full_name = ''
extracted_address = ''
extracted_id_number = ''
extracted_birth_date = ''
extracted_job_title = ''
extracted_workplace = ''
extracted_gender = ''
extracted_religion = ''
extracted_marital_status = ''
extracted_spouse_name = ''
extracted_governorate = ''

def extract_front_side_data(front_image_path):
    import cv2 as _cv2

    grayscale_image = _cv2.imread(front_image_path, _cv2.IMREAD_GRAYSCALE)
    if grayscale_image is None:
        raise ValueError(f"Could not load image from path: {front_image_path}")
    print(f"    [DEBUG] Front image shape: {grayscale_image.shape}")

    _, binary_image = _cv2.threshold(grayscale_image, 90, 255, _cv2.THRESH_BINARY)

    global extracted_photo, extracted_full_name, extracted_address
    global extracted_id_number, extracted_birth_date, extracted_governorate

    extracted_photo = grayscale_image[50:350, 50:275]
    name_region = binary_image[150:310, 400:1000]
    address_region = binary_image[300:450, 400:1000]

    h, w = grayscale_image.shape[:2]
    id_region_gray = grayscale_image[int(h*0.75):int(h*0.92), 0:w]
    id_adaptive = _cv2.adaptiveThreshold(
        id_region_gray, 255,
        _cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        _cv2.THRESH_BINARY, 11, 2
    )
    _, id_otsu = _cv2.threshold(id_region_gray, 0, 255, _cv2.THRESH_BINARY + _cv2.THRESH_OTSU)
    print(f"    [DEBUG] ID region shape: {id_region_gray.shape}")

    try:
        extracted_full_name = image_to_string(name_region, lang="ara")
        extracted_address = image_to_string(address_region, lang="ara")

        arabic_to_ascii = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
        extracted_id_number = ""

        for region_name, region in [("gray", id_region_gray), ("adaptive", id_adaptive), ("otsu", id_otsu)]:
            for lang, cfg in [
                ("ara_number", "--oem 1 --psm 7"),
                ("ara_number", "--oem 1 --psm 6"),
                ("eng",        "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789"),
                ("ara+eng",    "--oem 1 --psm 6"),
            ]:
                try:
                    result = image_to_string(region, lang=lang, config=cfg).strip()
                    digits_only = ''.join(c for c in result.translate(arabic_to_ascii) if c.isdigit())
                    print(f"    [DEBUG] {region_name}+{lang}: raw={repr(result[:60])} digits={digits_only}")
                    if len(digits_only) >= 10:
                        extracted_id_number = digits_only
                        break
                except Exception as e:
                    print(f"    [DEBUG] {region_name}+{lang} failed: {e}")
            if extracted_id_number:
                break

        print(f"Extracted ID Number: {extracted_id_number}")

    except Exception as ocr_error:
        print(f"[Warning] OCR extraction failed: {ocr_error}")
        return

    if len(extracted_id_number) >= 7:
        try:
            if extracted_id_number[0] == '2':
                birth_year = '19' + extracted_id_number[1:3]
            else:
                birth_year = '20' + extracted_id_number[1:3]
            birth_month = extracted_id_number[3:5]
            birth_day = extracted_id_number[5:7]
            extracted_birth_date = f"{birth_year}/{birth_month}/{birth_day}"
        except (IndexError, ValueError) as date_error:
            print(f"[Warning] Could not parse birth date from ID: {date_error}")
            extracted_birth_date = "Unknown"

    try:
        extracted_governorate = pob.placeOfBirth(extracted_id_number)
    except Exception as gov_error:
        print(f"[Warning] Could not determine governorate: {gov_error}")
        extracted_governorate = "Unknown"

def extract_and_validate_text_region(x1, y1, x2, y2, field_type, binary_image, full_text_context):
    """
    Extract and validate text from a specific region of the ID card image.

    This function crops a specific region, performs OCR, and applies field-specific
    validation and normalization for better accuracy.

    Args:
        x1, y1, x2, y2 (int): Coordinates defining the crop region (left, top, right, bottom)
        field_type (str): Type of field being extracted ('religion', 'gender', 'marital_status')
        binary_image (PIL.Image): Binary processed image for OCR
        full_text_context (str): Full OCR text from entire image for context validation

    Returns:
        str: Validated and normalized text for the specified field
    """
    # Define the crop area and extract the region
    crop_coordinates = (x1, y1, x2, y2)
    cropped_region = binary_image.crop(crop_coordinates)

    # Perform OCR on the cropped region
    try:
        extracted_text = image_to_string(cropped_region, lang='ara')
    except Exception as ocr_error:
        print(f"[Warning] OCR failed for {field_type}: {ocr_error}")
        return "Unknown"

    # Apply field-specific validation and normalization
    if field_type == 'religion':
        return _normalize_religion_text(full_text_context)
    elif field_type == 'gender':
        return _normalize_gender_text(full_text_context)
    elif field_type == 'marital_status':
        return _normalize_marital_status_text(full_text_context)
    else:
        # For other fields, return the raw extracted text
        return extracted_text.strip()


def _normalize_religion_text(full_text):
    """
    Normalize religion text based on common variations in Egyptian IDs.

    Args:
        full_text (str): Full OCR text context for validation

    Returns:
        str: Normalized religion text
    """
    religion_mappings = {
        'مسلم': 'مسلم',      # Muslim (male)
        'مسلمة': 'مسلمة',    # Muslim (female)
        'مسيحي': 'مسيحي',   # Christian (male)
        'مسيحى': 'مسيحي',   # Christian (male) - alternative spelling
        'مسيحية': 'مسيحية'  # Christian (female)
    }

    for variant, normalized in religion_mappings.items():
        if variant in full_text:
            return normalized

    return "غير محدد"  # Not specified


def _normalize_gender_text(full_text):
    """
    Normalize gender text based on common variations in Egyptian IDs.

    Args:
        full_text (str): Full OCR text context for validation

    Returns:
        str: Normalized gender text
    """
    if 'ذكر' in full_text:
        return 'ذكر'  # Male
    elif 'أنثى' in full_text or 'انثى' in full_text:
        return 'أنثى'  # Female

    return "غير محدد"  # Not specified


def _normalize_marital_status_text(full_text):
    """
    Normalize marital status text based on common variations in Egyptian IDs.

    Args:
        full_text (str): Full OCR text context for validation

    Returns:
        str: Normalized marital status text
    """
    marital_status_mappings = {
        'أنسة': 'أنسة',      # Miss/Single (female)
        'أعزب': 'أعزب',     # Single (male)
        'متزوج': 'متزوج',   # Married (male)
        'متزوجة': 'متزوجة'  # Married (female)
    }

    for variant, normalized in marital_status_mappings.items():
        if variant in full_text:
            return normalized

    return "غير محدد"  # Not specified


def extract_back_side_data(back_image_path):
    """
    Extract professional and personal information from the back side of Egyptian national ID card.

    This function processes the back side of an ID card to extract:
    - Job title/profession
    - Workplace/employer
    - Gender (derived from ID number)
    - Religion
    - Marital status
    - Spouse name (if applicable)

    Args:
        back_image_path (str): Path to the back side image of the ID card

    Returns:
        None: Updates global variables with extracted data

    Raises:
        ValueError: If image cannot be loaded
    """
    try:
        # Load the back side image using PIL and OpenCV
        pil_image = Image.open(back_image_path)  # Expected size: 1005x630
        grayscale_image = cv2.imread(back_image_path, cv2.IMREAD_GRAYSCALE)

        if grayscale_image is None:
            raise ValueError(f"Could not load image from path: {back_image_path}")

    except Exception as load_error:
        print(f"[Error] Failed to load back side image: {load_error}")
        return

    # Apply binary threshold for better OCR accuracy
    ocr_threshold_value = 145
    _, binary_thresholded_image = cv2.threshold(
        grayscale_image,
        ocr_threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    # Create morphological kernel for noise reduction (currently unused)
    morphological_kernel = np.ones((2, 2), np.uint8)

    # Save temporary thresholded image for PIL processing
    temp_threshold_path = 'threshold.jpg'
    cv2.imwrite(temp_threshold_path, binary_thresholded_image)

    try:
        # Load the thresholded image with PIL for cropping operations
        binary_pil_image = Image.open(temp_threshold_path)

        # Perform full-page OCR to get context for validation
        full_page_text = image_to_string(binary_pil_image, lang='ara')

    except Exception as ocr_error:
        print(f"[Error] Failed to perform OCR on back side: {ocr_error}")
        return

    # Access global variables for storing extracted data
    global extracted_job_title, extracted_workplace, extracted_gender
    global extracted_religion, extracted_marital_status, extracted_spouse_name
    global extracted_id_number

    # Extract job title from upper region
    extracted_job_title = extract_and_validate_text_region(
        230, 70, 820, 140,
        'job_title',
        binary_pil_image,
        full_page_text
    )

    # Extract workplace/employer information
    extracted_workplace = extract_and_validate_text_region(
        230, 125, 820, 190,
        'workplace',
        binary_pil_image,
        full_page_text
    )

    # Determine gender from ID number (more reliable than OCR)
    try:
        extracted_gender = gender_detector.gen(extracted_id_number)
    except Exception as gender_error:
        print(f"[Warning] Could not determine gender from ID: {gender_error}")
        # Fallback to OCR-based gender detection
        extracted_gender = extract_and_validate_text_region(
            700, 180, 820, 260,
            'gender',
            binary_pil_image,
            full_page_text
        )

    # Extract religion information
    extracted_religion = extract_and_validate_text_region(
        480, 180, 760, 260,
        'religion',
        binary_pil_image,
        full_page_text
    )

    # Extract marital status
    extracted_marital_status = extract_and_validate_text_region(
        200, 180, 570, 260,
        'marital_status',
        binary_pil_image,
        full_page_text
    )

    # Extract spouse name (relevant for married individuals)
    extracted_spouse_name = extract_and_validate_text_region(
        200, 225, 820, 290,
        'spouse_name',
        binary_pil_image,
        full_page_text
    )

    # Clean up temporary file
    try:
        os.remove(temp_threshold_path)
    except OSError:
        pass  # Ignore if file doesn't exist or can't be removed


def get_extracted_data_summary():
    """
    Get a comprehensive summary of all extracted ID card data.

    Returns:
        dict: Dictionary containing all extracted information from both sides
    """
    return {
        'personal_info': {
            'full_name': extracted_full_name,
            'address': extracted_address,
            'id_number': extracted_id_number,
            'birth_date': extracted_birth_date,
            'governorate': extracted_governorate
        },
        'professional_info': {
            'job_title': extracted_job_title,
            'workplace': extracted_workplace
        },
        'personal_details': {
            'gender': extracted_gender,
            'religion': extracted_religion,
            'marital_status': extracted_marital_status,
            'spouse_name': extracted_spouse_name
        }
    }


def reset_extracted_data():
    """
    Reset all global variables to empty state for processing new ID card.

    This function should be called before processing a new ID card to ensure
    no data from previous processing remains.
    """
    global extracted_photo, extracted_full_name, extracted_address
    global extracted_id_number, extracted_birth_date, extracted_job_title
    global extracted_workplace, extracted_gender, extracted_religion
    global extracted_marital_status, extracted_spouse_name, extracted_governorate

    extracted_photo = np.zeros((300, 225))
    extracted_full_name = ''
    extracted_address = ''
    extracted_id_number = ''
    extracted_birth_date = ''
    extracted_job_title = ''
    extracted_workplace = ''
    extracted_gender = ''
    extracted_religion = ''
    extracted_marital_status = ''
    extracted_spouse_name = ''
    extracted_governorate = ''


# Legacy function aliases for backward compatibility
def front_read(front_src):
    """
    Legacy function name for backward compatibility.

    Args:
        front_src (str): Path to front side image
    """
    return extract_front_side_data(front_src)


def back_read(source_image):
    """
    Legacy function name for backward compatibility.

    Args:
        source_image (str): Path to back side image
    """
    return extract_back_side_data(source_image)


def crop(dim1, dim2, dim3, dim4, name, img_binary, text_all):
    """
    Legacy function name for backward compatibility.

    Args:
        dim1, dim2, dim3, dim4: Crop coordinates
        name: Field type
        img_binary: Binary image
        text_all: Full text context

    Returns:
        str: Extracted and validated text
    """
    return extract_and_validate_text_region(dim1, dim2, dim3, dim4, name, img_binary, text_all)

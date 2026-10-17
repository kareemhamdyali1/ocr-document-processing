"""
Egyptian National ID Card OCR Data Extraction Module
"""

from PIL import Image
import cv2
import numpy as np
import os
from pytesseract import image_to_string
import pytesseract

import pob
import gender as gender_detector

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

    # تكبير منطقة الرقم x3 لتحسين الـ OCR
    id_region_gray = _cv2.resize(
        id_region_gray, None, fx=3, fy=3,
        interpolation=_cv2.INTER_CUBIC
    )

    id_adaptive = _cv2.adaptiveThreshold(
        id_region_gray, 255,
        _cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        _cv2.THRESH_BINARY, 11, 2
    )
    _, id_otsu = _cv2.threshold(
        id_region_gray, 0, 255,
        _cv2.THRESH_BINARY + _cv2.THRESH_OTSU
    )
    print(f"    [DEBUG] ID region shape after resize: {id_region_gray.shape}")

    try:
        extracted_full_name = image_to_string(name_region, lang="ara")
        extracted_address = image_to_string(address_region, lang="ara")

        arabic_to_ascii = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
        extracted_id_number = ""

        for region_name, region in [
            ("gray",     id_region_gray),
            ("adaptive", id_adaptive),
            ("otsu",     id_otsu),
        ]:
            for lang, cfg in [
                ("ara_number",   "--oem 1 --psm 7"),
                ("ara_number",   "--oem 1 --psm 6"),
                ("ara_combined", "--oem 1 --psm 7"),
                ("ara_combined", "--oem 1 --psm 6"),
                ("eng",          "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789"),
                ("ara+eng",      "--oem 1 --psm 6"),
            ]:
                try:
                    result = image_to_string(region, lang=lang, config=cfg).strip()
                    digits_only = ''.join(
                        c for c in result.translate(arabic_to_ascii) if c.isdigit()
                    )
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
    crop_coordinates = (x1, y1, x2, y2)
    cropped_region = binary_image.crop(crop_coordinates)
    try:
        extracted_text = image_to_string(cropped_region, lang='ara')
    except Exception as ocr_error:
        print(f"[Warning] OCR failed for {field_type}: {ocr_error}")
        return "Unknown"
    if field_type == 'religion':
        return _normalize_religion_text(full_text_context)
    elif field_type == 'gender':
        return _normalize_gender_text(full_text_context)
    elif field_type == 'marital_status':
        return _normalize_marital_status_text(full_text_context)
    else:
        return extracted_text.strip()


def _normalize_religion_text(full_text):
    religion_mappings = {
        'مسلم': 'مسلم',
        'مسلمة': 'مسلمة',
        'مسيحي': 'مسيحي',
        'مسيحى': 'مسيحي',
        'مسيحية': 'مسيحية',
    }
    for variant, normalized in religion_mappings.items():
        if variant in full_text:
            return normalized
    return "غير محدد"


def _normalize_gender_text(full_text):
    if 'ذكر' in full_text:
        return 'ذكر'
    elif 'أنثى' in full_text or 'انثى' in full_text:
        return 'أنثى'
    return "غير محدد"


def _normalize_marital_status_text(full_text):
    marital_status_mappings = {
        'أنسة': 'أنسة',
        'أعزب': 'أعزب',
        'متزوج': 'متزوج',
        'متزوجة': 'متزوجة',
    }
    for variant, normalized in marital_status_mappings.items():
        if variant in full_text:
            return normalized
    return "غير محدد"


def extract_back_side_data(back_image_path):
    try:
        pil_image = Image.open(back_image_path)
        grayscale_image = cv2.imread(back_image_path, cv2.IMREAD_GRAYSCALE)
        if grayscale_image is None:
            raise ValueError(f"Could not load image from path: {back_image_path}")
    except Exception as load_error:
        print(f"[Error] Failed to load back side image: {load_error}")
        return

    ocr_threshold_value = 145
    _, binary_thresholded_image = cv2.threshold(
        grayscale_image, ocr_threshold_value, 255, cv2.THRESH_BINARY
    )

    temp_threshold_path = 'threshold.jpg'
    cv2.imwrite(temp_threshold_path, binary_thresholded_image)

    try:
        binary_pil_image = Image.open(temp_threshold_path)
        full_page_text = image_to_string(binary_pil_image, lang='ara')
    except Exception as ocr_error:
        print(f"[Error] Failed to perform OCR on back side: {ocr_error}")
        return

    global extracted_job_title, extracted_workplace, extracted_gender
    global extracted_religion, extracted_marital_status, extracted_spouse_name
    global extracted_id_number

    extracted_job_title = extract_and_validate_text_region(
        230, 70, 820, 140, 'job_title', binary_pil_image, full_page_text
    )
    extracted_workplace = extract_and_validate_text_region(
        230, 125, 820, 190, 'workplace', binary_pil_image, full_page_text
    )

    try:
        extracted_gender = gender_detector.gen(extracted_id_number)
    except Exception as gender_error:
        print(f"[Warning] Could not determine gender from ID: {gender_error}")
        extracted_gender = extract_and_validate_text_region(
            700, 180, 820, 260, 'gender', binary_pil_image, full_page_text
        )

    extracted_religion = extract_and_validate_text_region(
        480, 180, 760, 260, 'religion', binary_pil_image, full_page_text
    )
    extracted_marital_status = extract_and_validate_text_region(
        200, 180, 570, 260, 'marital_status', binary_pil_image, full_page_text
    )
    extracted_spouse_name = extract_and_validate_text_region(
        200, 225, 820, 290, 'spouse_name', binary_pil_image, full_page_text
    )

    try:
        os.remove(temp_threshold_path)
    except OSError:
        pass


def get_extracted_data_summary():
    return {
        'personal_info': {
            'full_name': extracted_full_name,
            'address': extracted_address,
            'id_number': extracted_id_number,
            'birth_date': extracted_birth_date,
            'governorate': extracted_governorate,
        },
        'professional_info': {
            'job_title': extracted_job_title,
            'workplace': extracted_workplace,
        },
        'personal_details': {
            'gender': extracted_gender,
            'religion': extracted_religion,
            'marital_status': extracted_marital_status,
            'spouse_name': extracted_spouse_name,
        },
    }


def reset_extracted_data():
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


def front_read(front_src):
    return extract_front_side_data(front_src)


def back_read(source_image):
    return extract_back_side_data(source_image)


def crop(dim1, dim2, dim3, dim4, name, img_binary, text_all):
    return extract_and_validate_text_region(dim1, dim2, dim3, dim4, name, img_binary, text_all)

"""
Egyptian National ID Card OCR Data Extraction Module
"""
import re
from datetime import date
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


def _valid_egyptian_id(value):
    digits = re.sub(r"\D", "", str(value))
    if not re.fullmatch(r"[23]\d{13}", digits):
        return False

    try:
        century = 1900 if digits[0] == "2" else 2000
        year = century + int(digits[1:3])
        month = int(digits[3:5])
        day = int(digits[5:7])
        dob = date(year, month, day)

        valid_governorates = {
            "01", "02", "03", "04", "11", "12", "13",
            "14", "15", "16", "17", "18", "19", "21",
            "22", "23", "24", "25", "26", "27", "28",
            "29", "31", "32", "33", "34", "35", "88"
        }

        return dob <= date.today() and digits[7:9] in valid_governorates
    except ValueError:
        return False


def _read_id_region(grayscale_image):
    import cv2
    import pytesseract

    h, w = grayscale_image.shape[:2]

    # Coordinates relative to standardized 1000x630 card
    regions = [
        (450, 475, 975, 545),
        (400, 465, 990, 555),
        (350, 455, 995, 565),
    ]

    translation = str.maketrans(
        "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
        "01234567890123456789"
    )

    for region_index, (x1, y1, x2, y2) in enumerate(regions):
        x1, x2 = int(x1 * w / 1000), int(x2 * w / 1000)
        y1, y2 = int(y1 * h / 630), int(y2 * h / 630)

        crop = grayscale_image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        enlarged = cv2.resize(
            crop, None, fx=3, fy=3,
            interpolation=cv2.INTER_CUBIC
        )

        blurred = cv2.bilateralFilter(enlarged, 5, 35, 35)
        otsu = cv2.threshold(
            blurred, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        variants = [enlarged, blurred, otsu]

        for variant_index, variant in enumerate(variants):
            for lang in ("ara_number", "ara_combined", "ara", "eng"):
                for psm in (7, 13):
                    try:
                        raw = pytesseract.image_to_string(
                            variant,
                            lang=lang,
                            config=f"--oem 1 --psm {psm}"
                        )

                        normalized = raw.translate(translation)
                        digits = re.sub(r"\D", "", normalized)

                        # Never accept arbitrary long OCR output.
                        if len(digits) == 14 and _valid_egyptian_id(digits):
                            print(
                                "[national_id] Valid candidate found "
                                f"region={region_index} "
                                f"variant={variant_index} "
                                f"lang={lang} psm={psm}"
                            )
                            return digits

                    except Exception as exc:
                        print(
                            "[national_id] OCR attempt failed: "
                            f"{type(exc).__name__}"
                        )

    print("[national_id] No validated 14-digit ID found")
    return ""



def extract_front_side_data(front_image_path):
    import cv2 as _cv2

    global extracted_photo, extracted_full_name, extracted_address
    global extracted_id_number, extracted_birth_date, extracted_governorate

    grayscale_image = _cv2.imread(
        front_image_path, _cv2.IMREAD_GRAYSCALE
    )

    if grayscale_image is None:
        raise ValueError(
            f"Could not load image from path: {front_image_path}"
        )

    h, w = grayscale_image.shape[:2]

    print(
        f"[national_id] Front image loaded: "
        f"width={w}, height={h}"
    )

    # Reset front-side fields for each request
    extracted_full_name = ""
    extracted_address = ""
    extracted_id_number = ""
    extracted_birth_date = ""
    extracted_governorate = ""

    # Normalize card image to the repo's expected dimensions
    if w != 1000 or h != 630:
        grayscale_image = _cv2.resize(
            grayscale_image,
            (1000, 630),
            interpolation=_cv2.INTER_CUBIC
        )

    # Original repo preprocessing
    _, binary_image = _cv2.threshold(
        grayscale_image,
        90,
        255,
        _cv2.THRESH_BINARY
    )

    # Keep existing photo/name/address regions
    extracted_photo = grayscale_image[50:350, 50:275]

    name_region = binary_image[150:310, 400:1000]
    address_region = binary_image[300:450, 400:1000]

    # Extract name and address independently
    try:
        extracted_full_name = image_to_string(
            name_region,
            lang="ara",
            config="--oem 1 --psm 6"
        ).strip()

        print(
            "[national_id] Name OCR completed:",
            bool(extracted_full_name)
        )

    except Exception as exc:
        print(
            "[national_id] Name OCR failed:",
            type(exc).__name__
        )

    try:
        extracted_address = image_to_string(
            address_region,
            lang="ara",
            config="--oem 1 --psm 6"
        ).strip()

        print(
            "[national_id] Address OCR completed:",
            bool(extracted_address)
        )

    except Exception as exc:
        print(
            "[national_id] Address OCR failed:",
            type(exc).__name__
        )

    # Read only the ID-number area using multiple OCR variants
    try:
        extracted_id_number = _read_id_region(
            grayscale_image
        )

    except Exception as exc:
        extracted_id_number = ""

        print(
            "[national_id] ID region OCR failed:",
            type(exc).__name__
        )

    # Derive birth date and governorate only from validated ID
    if extracted_id_number and _valid_egyptian_id(
        extracted_id_number
    ):
        try:
            from datetime import date

            century = (
                1900
                if extracted_id_number[0] == "2"
                else 2000
            )

            year = century + int(
                extracted_id_number[1:3]
            )
            month = int(
                extracted_id_number[3:5]
            )
            day = int(
                extracted_id_number[5:7]
            )

            birth_date = date(year, month, day)

            extracted_birth_date = birth_date.strftime(
                "%Y/%m/%d"
            )

            try:
                extracted_governorate = pob.placeOfBirth(
                    extracted_id_number
                )

            except Exception as exc:
                extracted_governorate = ""

                print(
                    "[national_id] Governorate decoding failed:",
                    type(exc).__name__
                )

            print(
                "[national_id] ID validated successfully"
            )

        except Exception as exc:
            extracted_birth_date = ""
            extracted_governorate = ""

            print(
                "[national_id] ID decoding failed:",
                type(exc).__name__
            )

    else:
        extracted_id_number = ""
        extracted_birth_date = ""
        extracted_governorate = ""

        print(
            "[national_id] ID not validated; "
            "preserving name and address"
        )

    print(
        "[national_id] Front extraction completed:",
        {
            "full_name": bool(extracted_full_name),
            "address": bool(extracted_address),
            "national_id": bool(extracted_id_number),
            "date_of_birth": bool(extracted_birth_date),
            "governorate": bool(extracted_governorate),
        }
    )

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

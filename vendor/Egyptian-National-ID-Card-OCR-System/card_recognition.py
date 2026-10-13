"""
Egyptian National ID Card Scanner Module

This module provides functionality to scan and process Egyptian national ID cards.
It detects the card boundaries, applies perspective correction, and validates the ID number.

Main workflow:
1. Load and resize image for processing
2. Apply edge detection to find card boundaries
3. Find quadrilateral contours that could be the ID card
4. Validate candidates using OCR on the ID number region
5. Return the best processed card image
"""

from transform import apply_perspective_correction
from skimage.filters import threshold_local
from pytesseract import image_to_string
import numpy as np
import cv2
import imutils

# Configuration constants
PROCESSING_HEIGHT = 500          # Standard height for image processing
FINAL_CARD_SIZE = (1000, 630)    # Final output dimensions (width, height)
EXPECTED_ASPECT_RATIO = 1.58      # Expected ID card aspect ratio (W/H)
MIN_AREA_FRACTION = 0.05          # Minimum area as fraction of image (5%)
MAX_AREA_FRACTION = 0.65          # Maximum area as fraction of image (65%)
MAX_CONTOUR_CANDIDATES = 6        # Maximum number of contours to evaluate

def load_and_resize_image(image_path, target_height=PROCESSING_HEIGHT):
    """
    Load an image and resize it to a standard height while maintaining aspect ratio.

    Args:
        image_path (str): Path to the input image
        target_height (int): Target height for processing (default: 500px)

    Returns:
        tuple: (original_image, resized_image, scale_ratio)
            - original_image: Full resolution original image
            - resized_image: Resized image for processing
            - scale_ratio: Ratio to convert coordinates back to original size
    """
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Could not load image from path: {image_path}")

    height, width = original_image.shape[:2]
    scale_ratio = height / float(target_height)
    new_width = int(round(width / scale_ratio))

    resized_image = cv2.resize(
        original_image,
        (new_width, target_height),
        interpolation=cv2.INTER_AREA
    )

    return original_image.copy(), resized_image, scale_ratio

def preprocess_for_edge_detection(input_image):
    """
    Preprocess image for edge detection using adaptive Canny thresholds.

    Args:
        input_image: Input image (BGR format)

    Returns:
        tuple: (grayscale_image, edge_detected_image)
            - grayscale_image: Converted grayscale version
            - edge_detected_image: Binary edge map using Canny detection
    """
    # Convert to grayscale
    grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)

    # Calculate adaptive Canny thresholds based on image statistics
    median_pixel_intensity = np.median(blurred_image)
    canny_lower_threshold = int(max(0, 0.66 * median_pixel_intensity))
    canny_upper_threshold = int(min(255, 1.33 * median_pixel_intensity))

    # Apply Canny edge detection
    edge_detected_image = cv2.Canny(blurred_image, canny_lower_threshold, canny_upper_threshold)

    return grayscale_image, edge_detected_image


def extract_and_transform_card(card_corner_points, processed_image, full_resolution_image, coordinate_scale_ratio):
    """
    Extract the ID card region and apply perspective transformation.

    Args:
        card_corner_points: 4-point contour representing card corners
        processed_image: Processed image used for detection
        full_resolution_image: Full resolution original image
        coordinate_scale_ratio: Ratio to scale coordinates back to original size

    Returns:
        tuple: (debug_visualization, perspective_corrected_card, standardized_card_image)
            - debug_visualization: Image showing detected quadrilateral
            - perspective_corrected_card: Perspective-corrected card at original resolution
            - standardized_card_image: Resized card image at standard dimensions
    """
    # Create visualization image for debugging
    grayscale_for_viz = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    debug_visualization = cv2.cvtColor(grayscale_for_viz, cv2.COLOR_GRAY2BGR)

    if card_corner_points is None:
        return debug_visualization, None, None

    # Draw detected quadrilateral on visualization
    cv2.drawContours(
        debug_visualization,
        [card_corner_points.astype(int)],
        -1,
        (255, 0, 0),  # Blue color
        3
    )

    # Apply perspective transformation to original image
    full_resolution_corners = card_corner_points.reshape(4, 2) * coordinate_scale_ratio
    perspective_corrected_card = apply_perspective_correction(full_resolution_image, full_resolution_corners)

    # Resize to standard card dimensions
    standardized_card_image = cv2.resize(
        perspective_corrected_card,
        FINAL_CARD_SIZE,
        interpolation=cv2.INTER_LINEAR
    )

    return debug_visualization, perspective_corrected_card, standardized_card_image

def validate_id_number_region(standardized_card_image):
    """
    Extract and validate the ID number from the standardized card image.

    This function:
    1. Converts to grayscale and applies binary threshold
    2. Crops the expected ID number region
    3. Uses OCR to extract digits
    4. Normalizes Arabic digits to ASCII
    5. Validates minimum length requirement

    Args:
        standardized_card_image: Standardized card image (1000x630 pixels)

    Returns:
        tuple: (is_valid_id_number, extracted_digit_string)
            - is_valid_id_number: True if ID has >= 12 digits
            - extracted_digit_string: Normalized digit string
    """
    # Convert to grayscale and apply binary threshold
    grayscale_card = cv2.cvtColor(standardized_card_image, cv2.COLOR_BGR2GRAY)
    ocr_threshold_value = 90
    _, binary_thresholded_image = cv2.threshold(
        grayscale_card,
        ocr_threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    # Extract ID number region (empirically determined coordinates)
    id_number_region = binary_thresholded_image[480:560, 400:1000]

    # Perform OCR with Arabic number support
    try:
        # Try Arabic numbers language pack first
        raw_ocr_text = image_to_string(id_number_region, lang="ara_number")
    except Exception:
        # Fallback to Arabic+English with digit whitelist
        raw_ocr_text = image_to_string(
            id_number_region,
            lang="ara+eng",
            config="--psm 7 -c tessedit_char_whitelist=0123456789٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹"
        )

    # Normalize Arabic/Eastern Arabic digits to ASCII
    arabic_to_ascii_translation = str.maketrans(
        "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
        "01234567890123456789"
    )
    extracted_digit_string = "".join(
        character for character in raw_ocr_text.translate(arabic_to_ascii_translation)
        if character.isdigit()
    )

    # Validate minimum length (Egyptian IDs should have 14 digits)
    is_valid_id_number = len(extracted_digit_string) >= 12

    return is_valid_id_number, extracted_digit_string


def calculate_egyptian_id_confidence_score(id_number: str) -> float:
    """
    Calculate confidence score for Egyptian national ID number validity.

    Egyptian ID format: CYYMMDDSSGGG (14 digits)
    - C: Century (2=1900s, 3=2000s)
    - YY: Year of birth
    - MM: Month (01-12)
    - DD: Day (01-31)
    - SS: Sequence number
    - GGG: Governorate code + gender

    Scoring weights:
    - Century digit (2 or 3): 0.10
    - Valid month (01-12): 0.35
    - Valid day (01-31): 0.35
    - Valid governorate code: 0.20

    Args:
        id_number (str): Extracted ID number string

    Returns:
        float: Confidence score between 0.0 and 1.0
    """
    if not id_number or not id_number.isdigit():
        return 0.0

    confidence_score = 0.0

    # Validate century digit (should be 2 for 1900s or 3 for 2000s)
    if len(id_number) >= 1 and id_number[0] in ("2", "3"):
        confidence_score += 0.10

    # Validate month and day components
    if len(id_number) >= 7:
        try:
            month_str = id_number[3:5]
            day_str = id_number[5:7]

            if month_str.isdigit() and day_str.isdigit():
                month = int(month_str)
                day = int(day_str)

                # Valid month (01-12)
                if 1 <= month <= 12:
                    confidence_score += 0.35

                # Valid day (01-31) - basic validation
                if 1 <= day <= 31:
                    confidence_score += 0.35
        except (ValueError, IndexError):
            pass

    # Validate governorate code (positions 7-8)
    if len(id_number) >= 9:
        governorate_code = id_number[7:9]
        valid_governorate_codes = {
            "01", "02", "03", "04",  # Cairo, Alexandria, Port Said, Suez
            "11", "12", "13", "14", "15", "16", "17", "18", "19",  # Delta region
            "21", "22", "23", "24", "25", "26", "27", "28", "29",  # Upper Egypt
            "31", "32", "33", "34", "35",  # Frontier governorates
            "88"  # Born abroad
        }

        if governorate_code in valid_governorate_codes:
            confidence_score += 0.20

    return min(confidence_score, 1.0)

def find_best_card_contour(edge_detected_image, processed_image, full_resolution_image, coordinate_scale_ratio):
    """
    Find the best quadrilateral contour representing the ID card.

    This function:
    1. Finds all contours in the edge-detected image
    2. Filters for quadrilateral shapes (4 corners)
    3. Tests each candidate using OCR validation
    4. Returns the contour with the best ID number validation

    Args:
        edge_detected_image: Binary edge-detected image
        processed_image: Resized image used for processing
        full_resolution_image: Full resolution original image
        coordinate_scale_ratio: Scale factor for coordinate conversion

    Returns:
        numpy.ndarray: Best quadrilateral contour coordinates

    Raises:
        RuntimeError: If no suitable quadrilateral is found
    """
    # Find contours and sort by area (largest first)
    detected_contours = cv2.findContours(edge_detected_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    detected_contours = imutils.grab_contours(detected_contours)
    largest_contours = sorted(detected_contours, key=cv2.contourArea, reverse=True)[:5]

    best_validated_contour = None
    fallback_quadrilateral = None

    for current_contour in largest_contours:
        # Approximate contour to reduce number of points
        contour_perimeter = cv2.arcLength(current_contour, True)
        simplified_contour = cv2.approxPolyDP(current_contour, 0.02 * contour_perimeter, True)

        # Only consider quadrilaterals (4 corners)
        if len(simplified_contour) == 4:
            # Store first valid quadrilateral as fallback
            if fallback_quadrilateral is None:
                fallback_quadrilateral = simplified_contour

            try:
                # Extract and transform the card region
                _, _, candidate_card_image = extract_and_transform_card(
                    simplified_contour.reshape(4, 2),
                    processed_image,
                    full_resolution_image,
                    coordinate_scale_ratio
                )

                if candidate_card_image is not None:
                    # Validate using OCR on ID number region
                    is_valid_id_number, extracted_digit_string = validate_id_number_region(candidate_card_image)

                    # Calculate confidence score for additional validation
                    id_confidence_score = calculate_egyptian_id_confidence_score(extracted_digit_string)

                    # Log validation results for debugging
                    print(f"[ID Validation] digits='{extracted_digit_string}' "
                          f"length={len(extracted_digit_string)} | "
                          f"confidence={id_confidence_score:.2f}")

                    # Accept if ID validation passes
                    if is_valid_id_number:
                        best_validated_contour = simplified_contour
                        break

            except Exception as processing_error:
                # Continue to next contour if processing fails
                print(f"[Warning] Failed to process contour: {processing_error}")
                continue

    # Use fallback if no contour passed OCR validation
    if best_validated_contour is None and fallback_quadrilateral is not None:
        best_validated_contour = fallback_quadrilateral
        print("[Warning] Using fallback contour - OCR validation failed for all candidates")

    if best_validated_contour is None:
        raise RuntimeError(
            "No suitable quadrilateral contour found. "
            "Ensure the ID card is clearly visible and well-lit."
        )

    return best_validated_contour


def scan_id_card(input_image_path, card_side_type):
    """
    Main function to scan and process an Egyptian national ID card.

    This function performs the complete scanning workflow:
    1. Load and resize the input image
    2. Apply edge detection to find card boundaries
    3. Detect the best quadrilateral contour representing the card
    4. Apply perspective transformation to get a top-down view
    5. Save the processed card image

    Args:
        input_image_path (str): Path to the input image file
        card_side_type (str): Either "front" or "back" to specify card side

    Returns:
        numpy.ndarray: Processed card image (1000x630 pixels)

    Raises:
        ValueError: If image cannot be loaded
        RuntimeError: If no suitable card contour is found
    """
    # Step 1: Load and resize image for processing
    full_resolution_image, processing_sized_image, coordinate_scale_ratio = load_and_resize_image(
        input_image_path,
        PROCESSING_HEIGHT
    )

    # Step 2: Preprocess for edge detection
    grayscale_image, edge_detected_image = preprocess_for_edge_detection(processing_sized_image)

    # Step 3: Find the best card contour using OCR validation
    optimal_card_contour = find_best_card_contour(
        edge_detected_image,
        processing_sized_image,
        full_resolution_image,
        coordinate_scale_ratio
    )

    # Step 4: Apply perspective transformation
    full_resolution_corner_points = optimal_card_contour.reshape(4, 2) * coordinate_scale_ratio
    perspective_corrected_card = apply_perspective_correction(full_resolution_image, full_resolution_corner_points)

    # Step 5: Resize to standard dimensions and save
    final_processed_card = cv2.resize(
        perspective_corrected_card,
        FINAL_CARD_SIZE,
        interpolation=cv2.INTER_LINEAR
    )

    # Save processed image based on card side
    if card_side_type == "front":
        output_file_path = "temp_front.jpg"
    elif card_side_type == "back":
        output_file_path = "temp_back.jpg"
    else:
        raise ValueError(f"Invalid card_side_type: {card_side_type}. Must be 'front' or 'back'")

    cv2.imwrite(output_file_path, final_processed_card)
    print(f"[Success] Processed {card_side_type} side saved as {output_file_path}")

    return final_processed_card

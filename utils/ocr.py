# utils/ocr.py



import os

import re

import shutil

import cv2

import numpy as np

import pytesseract



from PIL import Image, ImageOps





# =========================================================

# Tesseract configuration

# =========================================================



# You can override the executable path using TESSERACT_CMD.

# Example in Windows PowerShell:

# $env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"




# Tesseract configuration: Windows and Railway/Linux

DEFAULT_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

TESSERACT_CMD = (
    os.environ.get("TESSERACT_CMD")
    or shutil.which("tesseract")
)

# Use the Windows fallback only on Windows.
if not TESSERACT_CMD and os.name == "nt":
    if os.path.isfile(DEFAULT_TESSERACT_PATH):
        TESSERACT_CMD = DEFAULT_TESSERACT_PATH

if not TESSERACT_CMD or not os.path.isfile(TESSERACT_CMD):
    raise RuntimeError(
        f"Tesseract executable not found. "
        f"OS={os.name}, TESSERACT_CMD={TESSERACT_CMD!r}"
    )

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

print(f"[OCR] Tesseract executable: {TESSERACT_CMD}")
print(
    f"[OCR] TESSDATA_PREFIX: "
    f"{os.environ.get('TESSDATA_PREFIX', '(not set)')}"
)





# =========================================================

# Image helpers

# =========================================================



def _to_rgb_array(pil_image: Image.Image) -> np.ndarray:

    """Normalize image orientation and convert to RGB."""

    if not isinstance(pil_image, Image.Image):

        raise TypeError("Expected a PIL Image.")



    image = ImageOps.exif_transpose(

        pil_image

    ).convert("RGB")



    array = np.asarray(image)



    if array.size == 0:

        raise ValueError("Input image is empty.")



    return array





def _upscale(gray: np.ndarray, target_size: int = 1600) -> np.ndarray:

    """Upscale small images without unnecessarily enlarging large scans."""

    height, width = gray.shape[:2]

    longest_side = max(height, width)



    if longest_side >= target_size:

        return gray



    scale = min(3.0, target_size / longest_side)



    new_width = max(1, int(width * scale))

    new_height = max(1, int(height * scale))



    return cv2.resize(

        gray,

        (new_width, new_height),

        interpolation=cv2.INTER_CUBIC,

    )





def preprocess(

    pil_image: Image.Image,

    aggressive: bool = False,

) -> np.ndarray:

    """

    Produce a grayscale OCR image.



    Keep grayscale as a candidate because thresholding can damage

    colored text, thin characters, stamps, or faint printing.

    """

    rgb = _to_rgb_array(pil_image)



    gray = cv2.cvtColor(

        rgb,

        cv2.COLOR_RGB2GRAY,

    )



    gray = _upscale(gray)



    if aggressive:

        gray = cv2.createCLAHE(

            clipLimit=2.5,

            tileGridSize=(8, 8),

        ).apply(gray)



        gray = cv2.fastNlMeansDenoising(

            gray,

            None,

            h=12,

            templateWindowSize=7,

            searchWindowSize=21,

        )

    else:

        gray = cv2.fastNlMeansDenoising(

            gray,

            None,

            h=8,

        )



    return gray





def preprocess_id_card(

    pil_image: Image.Image,

) -> list[np.ndarray]:

    """

    Create multiple image variants for national ID OCR.



    Returns grayscale, adaptive-threshold, Otsu, and sharpened variants.

    """

    rgb = _to_rgb_array(pil_image)



    height, width = rgb.shape[:2]

    longest_side = max(height, width)



    scale = min(

        4.0,

        max(1.0, 2000.0 / longest_side),

    )



    rgb = cv2.resize(

        rgb,

        (

            max(1, int(width * scale)),

            max(1, int(height * scale)),

        ),

        interpolation=cv2.INTER_CUBIC,

    )



    gray = cv2.cvtColor(

        rgb,

        cv2.COLOR_RGB2GRAY,

    )



    clahe = cv2.createCLAHE(

        clipLimit=2.0,

        tileGridSize=(8, 8),

    )



    enhanced = clahe.apply(gray)



    denoised = cv2.fastNlMeansDenoising(

        enhanced,

        None,

        h=10,

    )



    adaptive = cv2.adaptiveThreshold(

        denoised,

        255,

        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

        cv2.THRESH_BINARY,

        31,

        10,

    )



    blurred = cv2.GaussianBlur(

        gray,

        (3, 3),

        0,

    )



    _, otsu = cv2.threshold(

        blurred,

        0,

        255,

        cv2.THRESH_BINARY + cv2.THRESH_OTSU,

    )



    sharpening_kernel = np.array(

        [

            [0, -1, 0],

            [-1, 5, -1],

            [0, -1, 0],

        ],

        dtype=np.float32,

    )



    sharpened = cv2.filter2D(

        enhanced,

        -1,

        sharpening_kernel,

    )



    _, sharpened_otsu = cv2.threshold(

        sharpened,

        0,

        255,

        cv2.THRESH_BINARY + cv2.THRESH_OTSU,

    )



    return [

        gray,

        adaptive,

        otsu,

        sharpened_otsu,

    ]





# =========================================================

# Language and OCR helpers

# =========================================================




def _available_languages() -> set:
    """Return installed Tesseract languages and log discovery failures."""
    try:
        languages = set(pytesseract.get_languages(config=""))
        print(f"[OCR] Available Tesseract languages: {sorted(languages)}")
        return languages
    except Exception as exc:
        print(
            f"[OCR] Failed to detect Tesseract languages: "
            f"{type(exc).__name__}: {exc}"
        )
        return set()




def _select_language(requested_language: str) -> str:

    """

    Use the requested languages when installed.



    If ara is unavailable, use English if possible.

    """

    installed = _available_languages()



    requested = [

        part.strip()

        for part in requested_language.split("+")

        if part.strip()

    ]



    if requested and all(

        language in installed

        for language in requested

    ):

        return "+".join(requested)



    if "eng" in installed:

        return "eng"



    if requested and all(

        language in installed

        for language in requested

    ):

        return "+".join(requested)



    raise RuntimeError(

        "No suitable Tesseract language data found. "

        f"Installed languages: {sorted(installed)}. "

        "Install Arabic and English traineddata if required."

    )





def _run_tesseract(

    image: Image.Image,

    language: str,

    psm: int = 6,

) -> str:

    """Run Tesseract with a specified page segmentation mode."""

    config = f"--oem 3 --psm {psm}"



    text = pytesseract.image_to_string(

        image,

        lang=language,

        config=config,

    )



    return text.strip()





def _normalize_ocr_text(text: str) -> str:

    """Normalize line endings and remove empty duplicate whitespace."""

    text = text.replace("\r\n", "\n")

    text = text.replace("\r", "\n")

    text = text.replace("\x00", "")



    lines = [

        line.strip()

        for line in text.splitlines()

    ]



    return "\n".join(lines).strip()





def _text_quality(text: str) -> float:

    """

    Lightweight heuristic to rank OCR candidates.



    This is not an OCR confidence score and does not prove correctness.

    """

    if not text:

        return 0.0



    useful_characters = sum(

        1

        for char in text

        if char.isalnum()

        or "\u0600" <= char <= "\u06ff"

    )



    replacement_characters = text.count("\ufffd")



    repeated_noise = len(

        re.findall(

            r"(.)\1{5,}",

            text,

        )

    )



    return (

        useful_characters

        - 3.0 * replacement_characters

        - 2.0 * repeated_noise

    )





# =========================================================

# General OCR

# =========================================================



def ocr_image(

    pil_image: Image.Image,

    lang: str = "ara+eng",

) -> str:

    """

    Run OCR on several image variants and choose the best candidate

    using a basic text-quality heuristic.

    """

    selected_language = _select_language(lang)



    candidates = []



    variants = [

        preprocess(pil_image, aggressive=False),

        preprocess(pil_image, aggressive=True),

    ]



    # Include the original RGB image as a candidate.

    variants.append(_to_rgb_array(pil_image))



    for array in variants:

        if array.ndim == 2:

            candidate_image = Image.fromarray(array)

        else:

            candidate_image = Image.fromarray(array).convert("RGB")



        for psm in (6, 11):

            try:

                text = _run_tesseract(

                    candidate_image,

                    language=selected_language,

                    psm=psm,

                )



                text = _normalize_ocr_text(text)



                if text:

                    candidates.append(text)



            except pytesseract.TesseractError:

                continue



    if not candidates:

        raise RuntimeError(

            "Tesseract returned no usable text. "

            "Check the executable, traineddata, and input image."

        )



    return max(

        candidates,

        key=_text_quality,

    )





def ocr_pages(

    pages: list[Image.Image],

    lang: str = "ara+eng",

) -> str:

    """

    OCR every page and combine the text with page separators.



    Page failures are explicitly marked rather than silently ignored.

    """

    if not pages:

        raise ValueError(

            "No pages were supplied to ocr_pages()."

        )



    results = []



    for index, page in enumerate(pages, start=1):

        try:

            text = ocr_image(

                page,

                lang=lang,

            )



        except Exception as exc:

            text = (

                f"[OCR_ERROR: {type(exc).__name__}: {exc}]"

            )



        results.append(

            f"[PAGE {index}]\n{text}"

        )



    return "\n\n".join(results)





# =========================================================

# National ID OCR

# =========================================================



def ocr_id_card(

    pil_image: Image.Image,

    lang: str = "ara+eng",

) -> str:

    """

    Try several preprocessing variants and PSM settings.



    The ranking considers useful characters, Arabic characters,

    and digit presence. It is a heuristic, not a correctness guarantee.

    """

    selected_language = _select_language(lang)



    candidates = []



    for variant in preprocess_id_card(pil_image):

        candidate_image = Image.fromarray(variant)



        for psm in (6, 11, 7):

            try:

                text = _run_tesseract(

                    candidate_image,

                    language=selected_language,

                    psm=psm,

                )



                text = _normalize_ocr_text(text)



                if text:

                    candidates.append(text)



            except pytesseract.TesseractError:

                continue



    if not candidates:

        return ocr_image(

            pil_image,

            lang=lang,

        )



    def score(text):

        arabic_count = sum(

            "\u0600" <= char <= "\u06ff"

            for char in text

        )



        digit_count = sum(

            char.isdigit()

            for char in text

        )



        useful_count = sum(

            char.isalnum()

            or "\u0600" <= char <= "\u06ff"

            for char in text

        )



        return (

            useful_count

            + 1.5 * arabic_count

            + 2.0 * min(digit_count, 14)

        )



    return max(

        candidates,

        key=score,

    )

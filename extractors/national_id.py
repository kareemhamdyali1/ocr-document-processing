

"""Egyptian national ID extractor.
Primary: Marwan-Ghazal OCR repo.
Fallback: Tesseract + regex.
Refinement: enlarged front-card crops + Arabic Tesseract.
Recovery: multiple back-side OCR passes for profession and expiry date.
Optional fallback: Ollama vision.
Public API:

    extract(front_path: str, back_path: str) -> dict

"""



import contextlib
import io
import json
import os
import re
import sys
import base64
import urllib.request
from datetime import date, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


NATIONAL_ID_OCR_PATH = os.getenv(
    "NATIONAL_ID_OCR_PATH",
    str(
        Path(__file__).resolve().parent.parent
        / "vendor"
        / "Egyptian-National-ID-Card-OCR-System"
    ),
)


# NATIONAL_ID_OCR_PATH = os.getenv(

#     "NATIONAL_ID_OCR_PATH",

#     r"C:\Users\USER\OneDrive\Documents\OCR\files\Egyptian-National-ID-Card-OCR-System",

# )



# Crop coordinates on a normalized 1000 x 630 FRONT-card image.

NAME_CROP = (150, 290, 400, 1000)       # y1, y2, x1, x2
ADDRESS_CROP = (290, 450, 400, 1000)    # y1, y2, x1, x2
SAVE_DEBUG_CROPS = os.getenv("NATIONAL_ID_DEBUG_CROPS", "0") == "1"

GOVERNORATE_MAP = {

    "01": "القاهرة",
    "02": "الإسكندرية",
    "03": "بورسعيد",
    "04": "السويس",
    "11": "دمياط",
    "12": "الدقهلية",
    "13": "الشرقية",
    "14": "القليوبية",
    "15": "كفر الشيخ",
    "16": "الغربية",
    "17": "المنوفية",
    "18": "البحيرة",
    "19": "الإسماعيلية",
    "21": "الجيزة",
    "22": "بني سويف",
    "23": "الفيوم",
    "24": "المنيا",
    "25": "أسيوط",
    "26": "سوهاج",
    "27": "قنا",
    "28": "أسوان",
    "29": "الأقصر",
    "31": "البحر الأحمر",
    "32": "الوادي الجديد",
    "33": "مطروح",
    "34": "شمال سيناء",
    "35": "جنوب سيناء",
    "88": "خارج الجمهورية",

}

# ---------------------------------------------------------------------

# General utilities

# ---------------------------------------------------------------------



def _normalize_arabic_digits(value: str) -> str:
    return str(value or "").translate(
        str.maketrans(
            "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
            "01234567890123456789",
        )
    )


def _parse_id_number(nid: str) -> dict:
    """Decode date of birth, age, gender and governorate from a valid ID."""
    nid = re.sub(r"\D", "", _normalize_arabic_digits(nid))
    if len(nid) != 14 or nid[0] not in "23":
        return {}
    century = {"2": "19", "3": "20"}[nid[0]]
    try:
        dob = f"{century}{nid[1:3]}-{nid[3:5]}-{nid[5:7]}"
        dob_date = datetime.strptime(dob, "%Y-%m-%d").date()
        if dob_date > date.today():
            return {}
        age = round((date.today() - dob_date).days / 365.25, 1)
    except ValueError:
        return {}

    try:
        gender = "ذكر" if int(nid[12]) % 2 else "أنثى"
    except (ValueError, IndexError):
        gender = None



    return {
        "date_of_birth": dob,
        "age_years": age,
        "governorate": GOVERNORATE_MAP.get(nid[7:9]),
        "gender": gender,
    }






def _clean_field_text(value: str) -> str:
    """Clean OCR noise and fix Arabic-Indic digit misreads in addresses."""
    import re

    value = str(value or "")

    # Remove Unicode directional markers
    value = re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]", "", value)

    # Normalize Arabic-Indic digits to ASCII (٣٥٩ → 359)
    value = value.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))

    # Remove field labels
    value = re.split(
        r"(?:الرقم القومي|رقم البطاقة|تاريخ الميلاد|"
        r"تاريخ الانتهاء|المهنة|الوظيفة|العنوان|"
        r"محل الإقامة|محل الاقامة|البطاقة سارية حتي|الجنسية|"
        r"الحالة الاجتماعية|الديانة)",
        value, maxsplit=1, flags=re.IGNORECASE
    )[0]

    value = re.sub(r"\s+", " ", value).strip()
    value = value.strip(' :-|،؛,."\'«»""')

    # Fix common OCR misreads in addresses:
    # Arabic card often prints ٣٥٩ which Tesseract reads as "4 "" or "q"
    # We detect suspiciously short leading tokens and clean them
    value = re.sub(r'^[0-9]{1,2}\s*["\'"`]+\s*', '', value)

    return value.strip()





def _is_missing(value) -> bool:

    if value is None:

        return True

    if isinstance(value, str):

        return value.strip().lower() in {

            "",
            "unknown",
            "none",
            "null",
            "غير محدد",
            "غير معروف",
            "n/a",

        }



    return False





def _extract_labeled_field(

    text: str,

    labels: list[str],

    min_length: int = 4,

    max_length: int = 120,

):

    for line in (text or "").splitlines():

        for label in labels:

            match = re.search(

                rf"(?:{label})\s*[:：\-]?\s*(.+)",

                line,

                re.IGNORECASE,

            )



            if match:

                value = _clean_field_text(match.group(1))[:max_length]



                if len(value) >= min_length:

                    return value



    return None





# ---------------------------------------------------------------------

# Back-side OCR recovery

# ---------------------------------------------------------------------



def _ocr_back_variants(back_path: str) -> list[str]:

    """Run multiple OCR passes on the back-side image."""
    import cv2
    import numpy as np
    import pytesseract
    from utils.pdf_to_images import load_document

    pages = load_document(back_path)

    if not pages:
        return []

    page = pages[0]


    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



    # Support PIL images and OpenCV/NumPy images.

    if hasattr(page, "convert"):

        image = cv2.cvtColor(

            np.array(page.convert("RGB")),

            cv2.COLOR_RGB2BGR,

        )

    else:

        image = np.asarray(page).copy()

    if image.size == 0:

        return []



    if len(image.shape) == 3:

        if image.shape[2] == 4:

            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

        else:

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



    # Enlarge small images to improve recognition of fine text.

    if max(image.shape[:2]) < 1800:

        image = cv2.resize(

            image,

            None,

            fx=2,

            fy=2,

            interpolation=cv2.INTER_CUBIC,

        )



    variants = [image]



    denoised = cv2.fastNlMeansDenoising(

        image,

        None,

        10,

        7,

        21,

    )

    variants.append(denoised)



    thresholded = cv2.threshold(

        denoised,

        0,

        255,

        cv2.THRESH_BINARY + cv2.THRESH_OTSU,

    )[1]



    variants.append(thresholded)



    results = []

    seen = set()



    for variant in variants:

        for psm in (6, 11, 12):

            try:

                text = pytesseract.image_to_string(

                    variant,

                    lang="ara+eng",

                    config=f"--oem 1 --psm {psm}",

                )



                text = _normalize_arabic_digits(text or "").strip()



                if text and text not in seen:

                    seen.add(text)

                    results.append(text)



            except Exception as exc:

                print(

                    f"    [national_id] Back OCR PSM {psm} failed: "

                    f"{type(exc).__name__}"

                )



    print(

        f"    [national_id] Back-side OCR produced "

        f"{len(results)} text variants"

    )



    return results





# ---------------------------------------------------------------------

# Profession extraction

# ---------------------------------------------------------------------



def _extract_profession(texts: list) -> str | None:
    """
    Extract profession/qualification from back-side OCR variants.
    Joins continuation lines and cleans OCR noise.
    """
    import re

    label_pattern = (
        r"(?:المهنة|الوظيفة|المؤهل\s*الدراسي|المؤهل|"
        r"حاصل\s*على|profession|occupation|qualification)"
    )

    stop_pattern = (
        r"الرقم\s*القومي|رقم\s*البطاقة|تاريخ\s*الميلاد|"
        r"تاريخ\s*الانتهاء|البطاقة\s*سارية|العنوان|"
        r"محل\s*الإقامة|الجنسية|الديانة|الحالة\s*الاجتماعية|"
        r"expiry|valid\s*until|ذكر|أنثى|أعزب|متزوج"
    )

    candidates = []

    for text in texts:
        lines = [
            re.sub(r"\s+", " ", line).strip(" :-|،؛")
            for line in (text or "").splitlines()
        ]
        lines = [l for l in lines if l]

        for index, line in enumerate(lines):
            # Match label OR lines starting with حاصل على
            match = re.search(
                label_pattern + r"\s*[:：\-]?\s*(.*)",
                line,
                re.IGNORECASE,
            )

            if not match:
                # Also catch lines that start the profession directly
                if re.match(r"حاصل\s*على", line, re.IGNORECASE):
                    match_text = line
                else:
                    continue
            else:
                match_text = match.group(0)

            pieces = []
            # Include the matched line itself
            cleaned_first = re.sub(r"\s+", " ", match_text).strip(" :-|،؛")
            if len(cleaned_first) >= 4:
                pieces.append(cleaned_first)

            # Capture continuation lines (up to 6 more)
            for next_line in lines[index + 1: index + 7]:
                if re.search(stop_pattern, next_line, re.IGNORECASE):
                    break
                if re.search(label_pattern, next_line, re.IGNORECASE):
                    break
                # Skip lines that are mostly noise (short, lots of Latin chars)
                arabic_chars = len(re.findall(r"[\u0621-\u064A]", next_line))
                if arabic_chars < 2:
                    continue
                cleaned = re.sub(r"\s+", " ", next_line).strip(" :-|،؛")
                if cleaned and cleaned not in pieces:
                    pieces.append(cleaned)

            candidate = re.sub(r"\s+", " ", " ".join(pieces)).strip()

            if len(candidate) >= 10:
                candidates.append(candidate)

    if not candidates:
        return None

    # Return longest result (most complete)
    best = max(candidates, key=len)[:200]

    # Clean up OCR artifacts: remove isolated Latin chars and symbols
    # Remove OCR noise: isolated Latin chars, symbols, pipe chars, dashes
    best = re.sub(r"\b[A-Za-z$#@]{1,3}\b", "", best)
    best = re.sub(r"[|\\/{}<>]", "", best)
    best = re.sub(r"\s*—\s*$", "", best)
    best = re.sub(r"\s*كح\s*", " ", best)   # common OCR artifact
    best = re.sub(r"اتحاسبات", "الحاسبات", best)  # common misread
    best = re.sub(r"تخصّصض|تخصّص", "تخصص", best)  # normalize shadda noise
    best = re.sub(r"\s+", " ", best).strip(" :-|،؛")
    best = re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]", "", best)
    best = re.sub(r"[ؤ'،؛\|\\](?!\w)", " ", best)
    best = re.sub(r"\$\d+", "", best)      # remove $183 type noise
    best = re.sub(r"\s{2,}", " ", best).strip()

    return best if len(best) >= 10 else None



# ---------------------------------------------------------------------

# Expiry date extraction

# ---------------------------------------------------------------------



def _extract_expiry_date(texts: list) -> str | None:
    import unicodedata
    from datetime import date, datetime

    labels = (
        r"البطاقة\s*سارية\s*حتى",
        r"تاريخ\s*الانتهاء",
        r"صالح(?:ة)?\s*حتى",
        r"سارية?\s*حتى",
        r"expiry",
        r"valid\s*until",
    )

    date_pattern = re.compile(
        r"(?<!\d)(\d{1,4})\s*[/.\-]\s*"
        r"(\d{1,2})\s*[/.\-]\s*"
        r"(\d{1,4})(?!\d)"
    )

    def normalize(text):
        text = unicodedata.normalize("NFKC", str(text or ""))
        text = _normalize_arabic_digits(text)
        text = re.sub(
            r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]",
            "",
            text,
        )
        return re.sub(r"[ \t]+", " ", text)

    def parse_date(groups):
        a, b, c = groups
        candidates = []

        if len(a) == 4:
            candidates.append((int(a), int(b), int(c)))

        if len(c) == 4:
            candidates.append((int(c), int(b), int(a)))

        if len(a) == 2:
            candidates.append((2000 + int(a), int(b), int(c)))

        if len(c) == 2:
            candidates.append((2000 + int(c), int(b), int(a)))

        valid = []

        for year, month, day in candidates:
            try:
                parsed = date(year, month, day)
            except ValueError:
                continue

            if date.today() <= parsed <= date.today().replace(
                year=date.today().year + 20
            ):
                valid.append(parsed)

        # Avoid choosing arbitrarily between ambiguous formats.
        unique = set(valid)
        return next(iter(unique)).isoformat() if len(unique) == 1 else None

    for raw in texts:
        text = normalize(raw)

        for label in labels:
            for match in re.finditer(label, text, re.IGNORECASE):
                nearby = text[match.end():match.end() + 50]

                for dm in date_pattern.finditer(nearby):
                    result = parse_date(dm.groups())
                    if result:
                        return result

    return None




def _extract_expiry_with_qwen(back_path: str) -> str | None:
    """Read the expiry date using Qwen through the configured proxy."""
    import base64
    import io
    import re
    from datetime import date
    from pathlib import Path
    from PIL import Image
    from utils.ollama_client import ollama_generate

    image_path = Path("expiry_crop_enhanced.png")
    if not image_path.is_file():
        image_path = Path("expiry_crop.jpeg")
    if not image_path.is_file():
        image_path = Path(back_path)

    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image.thumbnail((1400, 800))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=95)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

        prompt = (
            "Inspect this Egyptian national ID back-side image. "
            "Find the date immediately associated with the Arabic label "
            "'البطاقة سارية حتى'. Return only JSON in this exact format: "
            '{"expiry_date":"YYYY-MM-DD"} or {"expiry_date":null}. '
            "Do not use the date of birth or another date. Never guess."
        )

        payload = {
            "model": os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b").strip(),
            "prompt": prompt,
            "images": [encoded],
            "stream": False,
            "keep_alive": "2m",
            "format": "json",
            "options": {
                "temperature": 0,
                "num_predict": 100,
                "num_ctx": 4096,
            },
        }

        body = ollama_generate(payload, timeout=180)
        raw = body.get("response", "").strip()

        if not raw:
            print("    [national_id] Qwen returned an empty expiry response")
            return None

        try:
            parsed = json.loads(raw)
            candidate = str(parsed.get("expiry_date") or "").strip()
        except (json.JSONDecodeError, AttributeError):
            candidate = raw

        # Accept only an unambiguous ISO date or a clear numeric date.
        candidate = candidate.translate(
            str.maketrans(
                "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
                "01234567890123456789",
            )
        )

        iso_match = re.search(r"(?<!\d)(20\d{2})-(\d{1,2})-(\d{1,2})(?!\d)", candidate)
        if iso_match:
            try:
                parsed_date = date(*map(int, iso_match.groups()))
                if date.today() <= parsed_date <= date.today().replace(
                    year=date.today().year + 20
                ):
                    return parsed_date.isoformat()
            except ValueError:
                pass

        numeric_match = re.search(
            r"(?<!\d)(\d{1,2})[/.-](\d{1,2})[/.-](20\d{2})(?!\d)",
            candidate,
        )
        if numeric_match:
            day, month, year = map(int, numeric_match.groups())
            try:
                parsed_date = date(year, month, day)
                if date.today() <= parsed_date <= date.today().replace(
                    year=date.today().year + 20
                ):
                    return parsed_date.isoformat()
            except ValueError:
                pass

        print("    [national_id] Qwen expiry response was not a valid future date")
        return None

    except Exception as exc:
        print(
            "    [national_id] Qwen expiry extraction failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return None

        def parse_date(text: str, require_label: bool) -> str | None:
            text = text.translate(digit_map)

            if require_label and not re.search(
                r"البطاقة\s*سارية\s*حتى", text
            ):
                return None

            if require_label:
                label = re.search(r"البطاقة\s*سارية\s*حتى", text)
                text = text[label.end():label.end() + 100]

            match = re.search(
                r"(?<!\d)(\d{1,4})\s*[/.\-]\s*"
                r"(\d{1,2})\s*[/.\-]\s*(\d{2,4})(?!\d)",
                text,
            )
            if not match:
                return None

            a, b, c = match.groups()
            try:
                # Supports DD/MM/YYYY and YYYY-MM-DD.
                if len(a) == 4:
                    parsed = date(int(a), int(b), int(c))
                else:
                    parsed = date(int(c), int(b), int(a))
                return parsed.isoformat()
            except ValueError:
                return None

        # First answer must explicitly contain the expiry label.
        first_date = parse_date(answers[0], require_label=True)

        # The second answer should be a date only.
        second_date = parse_date(answers[1], require_label=False)

        if first_date and second_date and first_date == second_date:
            print(f"    [national_id] Verified expiry date: {first_date}")
            return first_date

        print(
            "    [national_id] Expiry readings disagree or label "
            "was not detected; refusing to guess."
        )
        print(
            f"    [national_id] Parsed candidates: "
            f"{first_date!r}, {second_date!r}"
        )
        return None

    except Exception as exc:
        print(
            "    [national_id] Qwen expiry extraction failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return None


# ---------------------------------------------------------------------

# Primary OCR repository

# ---------------------------------------------------------------------



def _extract_via_repo(front_path: str, back_path: str) -> dict:

    if (

        not NATIONAL_ID_OCR_PATH

        or not Path(NATIONAL_ID_OCR_PATH).is_dir()

    ):

        raise FileNotFoundError(

            f"Repo path not found: {NATIONAL_ID_OCR_PATH}"

        )



    if NATIONAL_ID_OCR_PATH not in sys.path:

        sys.path.insert(0, NATIONAL_ID_OCR_PATH)



    import id_card_data_extractor as mod

    from card_recognition import scan_id_card



    fields = (

        "extracted_full_name",

        "extracted_address",

        "extracted_id_number",

        "extracted_birth_date",

        "extracted_governorate",

        "extracted_job_title",

        "extracted_workplace",

        "extracted_gender",

        "extracted_religion",

        "extracted_marital_status",

        "extracted_spouse_name",

    )



    output_stream = io.StringIO()



    with contextlib.redirect_stdout(output_stream):

        for field in fields:

            setattr(mod, field, "")



        scan_id_card(front_path, "front")

        scan_id_card(back_path, "back")



        mod.extract_front_side_data("temp_front.jpg")

        mod.extract_back_side_data("temp_back.jpg")



    output = output_stream.getvalue()



    data = {

        "full_name": getattr(mod, "extracted_full_name", None),

        "national_id": getattr(mod, "extracted_id_number", None),

        "date_of_birth": getattr(mod, "extracted_birth_date", None),

        "governorate": getattr(mod, "extracted_governorate", None),

        "address": getattr(mod, "extracted_address", None),

        "profession_on_id": getattr(mod, "extracted_job_title", None),

        "gender": getattr(mod, "extracted_gender", None),

        "marital_status_on_id": getattr(

            mod,

            "extracted_marital_status",

            None,

        ),

        # The upstream module may not expose an expiry field.

        "id_expiry_date": None,

    }



    validated = re.search(

        r"digits='(\d{14})'.*?confidence=1\.00",

        output,

        re.S,

    )



    if validated:

        data["national_id"] = validated.group(1)



    if not data.get("national_id"):

        found = re.search(

            r"Extracted ID Number:\s*(\d{14})",

            output,

        )



        if found:

            data["national_id"] = found.group(1)



    if not data.get("national_id"):

        raise ValueError(

            "Repo could not extract the national ID number"

        )



    for key, value in data.items():

        if isinstance(value, str):

            value = value.strip()



            if value.lower() in {

                "",

                "unknown",

                "غير محدد",

            }:

                value = None



            data[key] = value



    if data.get("date_of_birth"):

        data["date_of_birth"] = (

            data["date_of_birth"].replace("/", "-")

        )



    return data





# ---------------------------------------------------------------------

# Front-side crop refinement

# ---------------------------------------------------------------------



def _extract_name_address_from_image(front_path: str) -> dict:

    """Refine name/address using dedicated front-side image crops."""



    import cv2

    import pytesseract



    image = cv2.imread(

        str(front_path),

        cv2.IMREAD_GRAYSCALE,

    )



    if image is None:

        raise FileNotFoundError(

            f"Cannot read front image: {front_path}"

        )



    image = cv2.resize(

        image,

        (1000, 630),

        interpolation=cv2.INTER_CUBIC,

    )



    results = {

        "full_name": None,

        "address": None,

    }



    for field, bounds in (

        ("full_name", NAME_CROP),

        ("address", ADDRESS_CROP),

    ):

        y1, y2, x1, x2 = bounds



        if not (

            0 <= y1 < y2 <= 630

            and 0 <= x1 < x2 <= 1000

        ):

            raise ValueError(

                f"Invalid {field} crop: {bounds}"

            )



        region = image[y1:y2, x1:x2]



        enlarged = cv2.resize(

            region,

            None,

            fx=3,

            fy=3,

            interpolation=cv2.INTER_CUBIC,

        )



        if SAVE_DEBUG_CROPS:

            cv2.imwrite(

                f"debug_{field}_crop.png",

                enlarged,

            )



        # PSM 11: sparse text.

        # PSM 6: block of text.

        # PSM 7: single text line.

        raw_results = []



        for psm in (11, 6, 7):

            try:

                raw = pytesseract.image_to_string(

                    enlarged,

                    lang="ara",

                    config=f"--oem 1 --psm {psm}",

                )



                raw_results.append(raw or "")



            except Exception as exc:

                print(

                    f"    [national_id] {field} PSM {psm} failed: "

                    f"{type(exc).__name__}"

                )



        lines = []



        for raw in raw_results:

            for line in raw.splitlines():

                cleaned = _clean_field_text(line)



                if cleaned and cleaned not in lines:

                    lines.append(cleaned)



        if field == "full_name":

            candidates = []



            for line in lines:

                if re.search(r"\d", line):

                    continue



                words = re.findall(

                    r"[\u0621-\u064A]+",

                    line,

                )



                if 1 <= len(words) <= 5:

                    candidate = " ".join(words)



                    if len(candidate) >= 3:

                        candidates.append(candidate)



            # Select a plausible multi-word name from the crop.

            combined = " ".join(candidates[:2]).strip()

            word_count = len(combined.split())



            if 3 <= word_count <= 6:

                results[field] = combined



        else:

            # Preserve numbers in addresses.

            candidates = [

                line for line in lines

                if len(line) >= 3

                and len(

                    re.findall(r"[\u0621-\u064A]", line)

                ) >= 2

            ]



            candidate = " ".join(candidates[:6]).strip()

            # Fix leading OCR noise (e.g. '4 "' instead of '٣٥٩')
            # Keep leading building numbers (e.g. 359) but remove OCR quote noise
            candidate = re.sub(r'^([0-9]{1,4})?\s*["\'\u201c\u201d`]+\s*', r'\1 ', candidate)
            candidate = re.sub(r'^\s+', '', candidate)
            candidate = candidate.strip()



            if (

                len(candidate) >= 10

                and len(

                    re.findall(r"[\u0621-\u064A]", candidate)

                ) >= 6

            ):

                results[field] = candidate



    return results





# ---------------------------------------------------------------------

# Tesseract fallback

# ---------------------------------------------------------------------



def _extract_via_tesseract(

    front_path: str,

    back_path: str,

) -> dict:

    from utils.pdf_to_images import load_document

    from utils.ocr import ocr_id_card



    front_pages = load_document(front_path)

    back_pages = load_document(back_path)



    if not front_pages or not back_pages:

        raise ValueError(

            "National ID front/back image missing or unreadable"

        )



    front_text = _normalize_arabic_digits(

        ocr_id_card(

            front_pages[0],

            lang="ara+eng",

        ) or ""

    )



    back_text = _normalize_arabic_digits(

        ocr_id_card(

            back_pages[0],

            lang="ara+eng",

        ) or ""

    )



    back_variants = _ocr_back_variants(back_path)

    all_back_text = [back_text] + back_variants



    full_text = (

        front_text + "\n" + "\n".join(all_back_text)

    )



    match = re.search(

        r"(?<!\d)[23]\d{13}(?!\d)",

        full_text,

    )



    nid = match.group(0) if match else None



    name = _extract_labeled_field(

        front_text,

        [

            r"الاسم\s+رباعي(?:اً|ا)?",

            r"الاسم\s+بالكامل",

            r"الاسم",

            r"\bName\b",

        ],

        5,

        70,

    )



    address = _extract_labeled_field(

        front_text,

        [

            r"عنوان\s+الإقامة",

            r"عنوان\s+الاقامة",

            r"محل\s+الإقامة",

            r"محل\s+الاقامة",

            r"العنوان",

            r"\bAddress\b",

        ],

        5,

        150,

    )



    profession = _extract_profession(all_back_text)

    expiry = _extract_expiry_date(all_back_text)



    marital = None



    for pattern, status in (

        (r"أعزب|عزباء", "أعزب"),

        (r"متزوج|متزوجة", "متزوج"),

        (r"مطلق|مطلقة", "مطلق"),

        (r"أرمل|أرملة", "أرمل"),

    ):

        if re.search(pattern, full_text):

            marital = status

            break



    result = {

        "full_name": name,

        "national_id": nid,

        "address": address,

        "id_expiry_date": expiry,

        "marital_status_on_id": marital,

        "profession_on_id": profession,

    }



    if nid and len(nid) == 14:

        result.update(_parse_id_number(nid))



    return result





# ---------------------------------------------------------------------

# Optional Ollama vision fallback

# ---------------------------------------------------------------------




def _extract_via_ollama_vision(
    front_path: str,
    back_path: str,
) -> dict:
    """Extract ID fields using the configured remote/local vision model."""
    import base64
    from utils.ollama_client import ollama_generate

    def encode(path: str) -> str:
        with open(path, "rb") as handle:
            return base64.b64encode(handle.read()).decode("ascii")

    prompt = """
You are extracting fields from the front and back of an Egyptian national ID.
Return only a JSON object with these keys:
full_name, national_id, date_of_birth, governorate, address, gender,
marital_status_on_id, profession_on_id, id_expiry_date.

Rules:
- Use Arabic for Arabic text.
- Dates must be YYYY-MM-DD.
- Return null for unreadable or absent values.
- Never guess or invent information.
- national_id must contain exactly 14 digits.
- id_expiry_date is the future expiry date printed on the back,
  not date of birth or issue date.
- Read the full visible address and qualification when possible.
"""

    payload = {
        "model": os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b").strip(),
        "prompt": prompt,
        "images": [encode(front_path), encode(back_path)],
        "stream": False,
        "format": "json",
        "keep_alive": "2m",
        "options": {
            "temperature": 0,
            "num_predict": 700,
            "num_ctx": 4096,
        },
    }

    body = ollama_generate(payload, timeout=300)
    raw = body.get("response", "").strip()

    if not raw:
        raise ValueError("Qwen returned an empty vision response")

    raw = re.sub(r"^\s*```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```\s*$", "", raw).strip()
    result = json.loads(raw)

    if not isinstance(result, dict):
        raise ValueError("Qwen did not return a JSON object")

    print("    [national_id] Qwen vision extraction succeeded")
    return result




# ---------------------------------------------------------------------

# Main public function

# ---------------------------------------------------------------------



def extract(front_path: str, back_path: str) -> dict:

    """Extract ID fields, recover missing fields, and decode ID metadata."""



    repo_succeeded = False



    # 1. Primary extraction.

    try:

        print(

            "    [national_id] Trying Marwan-Ghazal OCR repo..."

        )



        data = _extract_via_repo(

            front_path,

            back_path,

        )



        repo_succeeded = True



        print(

            "    [national_id] Repo succeeded"

        )



    except Exception as exc:

        print(

            "    [national_id] Repo failed "

            f"({type(exc).__name__}: {exc}); "

            "trying Tesseract"

        )



        data = _extract_via_tesseract(

            front_path,

            back_path,

        )



        print(

            "    [national_id] Tesseract fallback done"

        )



    # Supplement missing fields with Tesseract even when the repository succeeds.
    # Do not overwrite existing non-empty values blindly.
    if repo_succeeded and any(_is_missing(data.get(k)) for k in (
        "full_name", "address", "profession_on_id", "id_expiry_date",
        "marital_status_on_id",
    )):
        try:
            supplemental = _extract_via_tesseract(front_path, back_path)
            for key, value in supplemental.items():
                if _is_missing(data.get(key)) and not _is_missing(value):
                    data[key] = value
            print("    [national_id] Tesseract supplemental recovery completed")
        except Exception as exc:
            print(f"    [national_id] Supplemental OCR failed: {exc}")

    # Ensure consistent output keys.

    for key in (

        "full_name",

        "national_id",

        "date_of_birth",

        "governorate",

        "address",

        "gender",

        "marital_status_on_id",

        "profession_on_id",

        "id_expiry_date",

    ):

        data.setdefault(key, None)



    # 2. Refine front-side name/address.

    # Use temp_front.jpg only if it was generated during this run.

    refinement_source = (

        "temp_front.jpg"

        if (

            repo_succeeded

            and Path("temp_front.jpg").is_file()

        )

        else front_path

    )



    try:

        refined = _extract_name_address_from_image(

            refinement_source

        )



        for field in ("full_name", "address"):

            if refined.get(field):

                data[field] = refined[field]



        print(

            "    [national_id] Name/address crop "

            "refinement completed"

        )



    except Exception as exc:

        print(

            "    [national_id] Name/address refinement failed: "

            f"{type(exc).__name__}: {exc}"

        )



    # 3. Recover profession and expiry from multiple back-side OCR passes,

    # even when the primary repository has already succeeded.


    # 3. Recover profession and expiry from back-side OCR.
    back_texts = []

    try:
        back_texts = _ocr_back_variants(back_path)

        if _is_missing(data.get("id_expiry_date")):
            recovered_expiry = _extract_expiry_date(back_texts)

            if recovered_expiry:
                data["id_expiry_date"] = recovered_expiry
                print(
                    "    [national_id] Expiry date recovered by OCR"
                )

        current_profession = data.get("profession_on_id")
        recovered_profession = _extract_profession(back_texts)

        if recovered_profession:
            current_text = (
                str(current_profession).strip()
                if not _is_missing(current_profession)
                else ""
            )

            if (
                not current_text
                or len(recovered_profession) > len(current_text)
            ):
                data["profession_on_id"] = recovered_profession
                print(
                    "    [national_id] Profession recovered/refined"
                )

    except Exception as exc:
        print(
            "    [national_id] Back-side recovery failed: "
            f"{type(exc).__name__}: {exc}"
        )

    # Qwen runs only if OCR did not recover a valid expiry date.
    if _is_missing(data.get("id_expiry_date")):
        qwen_expiry = _extract_expiry_with_qwen(back_path)

        if qwen_expiry:
            data["id_expiry_date"] = qwen_expiry
            print(
                "    [national_id] Expiry date recovered by Qwen"
            )
        else:
            data["id_expiry_date"] = None
            print(
                "    [national_id] Expiry date remains unreadable"
            )




    # 4. Optional vision fallback for missing/weak fields.

    placeholders = {

        "arabic full name",

        "full name",

        "yyyy-mm-dd",

        "governorate in arabic",

        "full address",

        "profession text",

        "name",

        "address",

        "profession",

        "marital status",

        "unknown",

        "غير محدد",

    }



    def missing(value):

        if _is_missing(value):

            return True



        return (

            isinstance(value, str)

            and value.strip().lower() in placeholders

        )



    vision_fields = (

        "full_name",

        "national_id",

        "address",

        "profession_on_id",

        "id_expiry_date",

    )



    if (
        os.getenv("NATIONAL_ID_USE_LEGACY_VISION", "0") == "1"
        and any(
            missing(data.get(field))
            for field in vision_fields
        )
    ):

        try:

            vision_data = _extract_via_ollama_vision(

                front_path,

                back_path,

            )



            for key in (

                "full_name",

                "national_id",

                "date_of_birth",

                "governorate",

                "address",

                "gender",

                "marital_status_on_id",

                "profession_on_id",

                "id_expiry_date",

            ):

                value = vision_data.get(key)



                if missing(value):

                    continue



                # Replace missing values only. Do not overwrite an

                # existing non-empty result with a vision guess.

                if missing(data.get(key)):

                    data[key] = value



        except Exception as exc:

            print(

                "    [national_id] Ollama vision unavailable: "

                f"{type(exc).__name__}: {exc}"

            )



    # 5. Normalize and validate national ID.

    nid = next(

        (

            data.get(key)

            for key in (

                "national_id",

                "id_number",

                "ID Number",

                "id",

                "National ID",

                "national_id_number",

            )

            if data.get(key)

        ),

        None,

    )



    if nid:

        nid = re.sub(

            r"\s+",

            "",

            _normalize_arabic_digits(str(nid)),

        )



        data["national_id"] = nid



        if re.fullmatch(r"[23]\d{13}", nid):

            decoded = _parse_id_number(nid)



            if decoded:

                # Derived ID metadata is based on the validated ID.

                data.update(decoded)



        else:

            print(

                "    [national_id] Warning: extracted ID "

                "does not match the expected 14-digit format"

            )



    # 6. Normalize common date representations.

    for field in (

        "date_of_birth",

        "id_expiry_date",

    ):

        value = data.get(field)



        if not isinstance(value, str):

            continue



        value = value.strip().replace("/", "-")



        # Normalize valid ISO dates only. Avoid silently interpreting

        # ambiguous day/month dates.

        try:

            parsed = datetime.strptime(

                value,

                "%Y-%m-%d",

            ).date()



            data[field] = parsed.isoformat()



        except ValueError:

            # Keep the original OCR value for inspection instead of

            # fabricating a date.

            data[field] = value or None



    # Always include all expected keys.

    data.setdefault("id_expiry_date", None)

    data.setdefault("profession_on_id", None)

    data.setdefault("address", None)



    return data


def extract_to_json(front_path: str, back_path: str, output_path: str = "national_id_result.json") -> dict:
    """Run extraction and save all available fields as UTF-8 JSON."""
    data = extract(front_path, back_path)
    Path(output_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    [national_id] JSON saved to: {Path(output_path).resolve()}")
    return data

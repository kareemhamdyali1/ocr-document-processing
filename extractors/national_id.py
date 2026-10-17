"""Egyptian national ID extractor.

Primary: Marwan-Ghazal OCR repository.
Fallback: Tesseract + regex.
Refinement: front-side name/address crops and multiple back-side OCR passes.
Optional recovery: configured Ollama/Qwen vision endpoint via utils.ollama_client.

Public API:
    extract(front_path: str, back_path: str) -> dict
    extract_to_json(front_path: str, back_path: str, output_path: str) -> dict
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NATIONAL_ID_OCR_PATH = os.getenv(
    "NATIONAL_ID_OCR_PATH",
    str(PROJECT_ROOT / "vendor" / "Egyptian-National-ID-Card-OCR-System"),
)
SAVE_DEBUG_CROPS = os.getenv("NATIONAL_ID_DEBUG_CROPS", "0") == "1"

NAME_CROP = (150, 290, 400, 1000)
ADDRESS_CROP = (290, 450, 400, 1000)

ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹"
ASCII_DIGITS = "01234567890123456789"
DIGIT_TRANSLATION = str.maketrans(ARABIC_DIGITS, ASCII_DIGITS)
DIRECTIONAL_MARKS = r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]"

GOVERNORATE_MAP = {
    "01": "القاهرة", "02": "الإسكندرية", "03": "بورسعيد", "04": "السويس",
    "11": "دمياط", "12": "الدقهلية", "13": "الشرقية", "14": "القليوبية",
    "15": "كفر الشيخ", "16": "الغربية", "17": "المنوفية", "18": "البحيرة",
    "19": "الإسماعيلية", "21": "الجيزة", "22": "بني سويف", "23": "الفيوم",
    "24": "المنيا", "25": "أسيوط", "26": "سوهاج", "27": "قنا",
    "28": "أسوان", "29": "الأقصر", "31": "البحر الأحمر", "32": "الوادي الجديد",
    "33": "مطروح", "34": "شمال سيناء", "35": "جنوب سيناء", "88": "خارج الجمهورية",
}

EXPECTED_KEYS = (
    "full_name", "national_id", "date_of_birth", "governorate", "address", "gender",
    "marital_status_on_id", "profession_on_id", "id_expiry_date",
)


def _normalize_digits(value: Any) -> str:
    return str(value or "").translate(DIGIT_TRANSLATION)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "unknown", "none", "null", "غير محدد", "غير معروف", "n/a"}
    return False


def _clean_text(value: Any) -> str:
    value = _normalize_digits(value)
    value = re.sub(DIRECTIONAL_MARKS, "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value.strip(" :-|،؛,.\"'«»")


def _find_national_id(texts: Any) -> str | None:
    if isinstance(texts, str):
        texts = [texts]
    candidate_pattern = re.compile(
        r"(?<![0-9])[23](?:[ \t.\-_/|:،؛]*[0-9]){13}(?![0-9])"
    )
    seen = set()

    def _search_line(line: str) -> str | None:
        # جرب الـ line عادي وجرب معكوسه (RTL OCR issue)
        for attempt in [line, line[::-1]]:
            for match in candidate_pattern.finditer(attempt):
                candidate = re.sub(r"\D", "", match.group(0))
                if len(candidate) != 14 or candidate in seen:
                    continue
                seen.add(candidate)
                decoded = _parse_id_number(candidate)
                if decoded.get("date_of_birth"):
                    return candidate
        return None

    for text in texts or []:
        normalized = _normalize_digits(text)
        normalized = re.sub(DIRECTIONAL_MARKS, "", normalized)
        for line in normalized.splitlines():
            result = _search_line(line)
            if result:
                return result
    return None


def _parse_id_number(nid: str) -> dict:
    nid = re.sub(r"\D", "", _normalize_digits(nid))
    if not re.fullmatch(r"[23]\d{13}", nid):
        return {}
    century = "19" if nid[0] == "2" else "20"
    try:
        dob_date = datetime.strptime(f"{century}{nid[1:7]}", "%Y%m%d").date()
        if dob_date > date.today():
            return {}
    except ValueError:
        return {}
    gender = "ذكر" if int(nid[12]) % 2 else "أنثى"
    return {
        "date_of_birth": dob_date.isoformat(),
        "age_years": round((date.today() - dob_date).days / 365.25, 1),
        "governorate": GOVERNORATE_MAP.get(nid[7:9]),
        "gender": gender,
    }


def _read_pages(path: str):
    from utils.pdf_to_images import load_document
    pages = load_document(path)
    if not pages:
        raise ValueError(f"Could not read document: {path}")
    return pages


def _to_gray_image(page):
    import cv2
    import numpy as np
    if hasattr(page, "convert"):
        arr = np.array(page.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    arr = np.asarray(page).copy()
    if arr.size == 0:
        raise ValueError("Image is empty")
    if arr.ndim == 2:
        return arr
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)


def _deskew_and_orient(gray):
    """Fix rotation using Tesseract OSD."""
    import cv2
    import pytesseract
    import numpy as np

    if max(gray.shape[:2]) < 1000:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    try:
        osd = pytesseract.image_to_osd(gray, config="--psm 0 -c min_characters_to_try=5")
        match = re.search(r"Rotate:\s*(\d+)", osd)
        if match:
            angle = int(match.group(1))
            if angle != 0:
                print(f"    [national_id] Auto-rotating image by {angle} degrees")
                h, w = gray.shape[:2]
                center = (w // 2, h // 2)
                matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
                gray = cv2.warpAffine(gray, matrix, (w, h),
                                       flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_REPLICATE)
    except Exception as exc:
        print(f"    [national_id] OSD skipped: {exc}")

    return gray


def _tesseract_text(image, lang="ara+eng", psm=6) -> str:
    import pytesseract
    return pytesseract.image_to_string(image, lang=lang, config=f"--oem 1 --psm {psm}") or ""


def _ocr_back_variants(back_path: str) -> list[str]:
    import cv2
    gray = _to_gray_image(_read_pages(back_path)[0])
    gray = _deskew_and_orient(gray)
    if max(gray.shape[:2]) < 1800:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    variants = [gray]
    try:
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        variants.append(denoised)
        variants.append(cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
    except Exception as exc:
        print(f"    [national_id] Back-image preprocessing warning: {type(exc).__name__}")
    results, seen = [], set()
    for variant in variants:
        for psm in (6, 11, 12):
            try:
                text = _normalize_digits(_tesseract_text(variant, "ara+eng", psm) or "")
                text = re.sub(DIRECTIONAL_MARKS, "", text).strip()
                if text and text not in seen:
                    seen.add(text)
                    results.append(text)
            except Exception as exc:
                print(f"    [national_id] Back OCR PSM {psm} failed: {type(exc).__name__}")
    print(f"    [national_id] Back-side OCR produced {len(results)} text variants")
    return results


def _extract_labeled_field(text: str, labels: list[str], min_length=4, max_length=150):
    for line in (text or "").splitlines():
        for label in labels:
            match = re.search(rf"(?:{label})\s*[:：\-]?\s*(.+)", line, re.IGNORECASE)
            if match:
                value = _clean_text(match.group(1))[:max_length]
                if len(value) >= min_length:
                    return value
    return None


def _extract_profession(texts: list[str]) -> str | None:
    label_pattern = r"(?:المهنة|الوظيفة|المؤهل\s*الدراسي|المؤهل|حاصل\s*على|profession|occupation|qualification)"
    stop_pattern = r"الرقم\s*القومي|رقم\s*البطاقة|تاريخ\s*الميلاد|تاريخ\s*الانتهاء|البطاقة\s*سارية|العنوان|محل\s*الإقامة|الجنسية|الديانة|الحالة\s*الاجتماعية|expiry|valid\s*until|ذكر|أنثى|أعزب|متزوج"
    candidates = []
    for text in texts:
        lines = [_clean_text(line) for line in (text or "").splitlines()]
        lines = [line for line in lines if line]
        for i, line in enumerate(lines):
            match = re.search(label_pattern + r"\s*[:：\-]?\s*(.*)", line, re.IGNORECASE)
            if not match and not re.match(r"حاصل\s*على", line):
                continue
            first = _clean_text(match.group(1) if match else line)
            pieces = [first] if len(first) >= 3 else []
            for nxt in lines[i + 1:i + 5]:
                if re.search(stop_pattern, nxt, re.IGNORECASE) or re.search(label_pattern, nxt, re.IGNORECASE):
                    break
                if len(re.findall(r"[\u0621-\u064a]", nxt)) >= 2:
                    pieces.append(nxt)
            candidate = _clean_text(" ".join(pieces))
            candidate = re.sub(r"\b[A-Za-z$#@]{1,3}\b|[|\\/{}<>]", " ", candidate)
            candidate = re.sub(r"\s+", " ", candidate).strip(" :-|،؛")
            if len(candidate) >= 5:
                candidates.append(candidate)
    if not candidates:
        return None
    return max(candidates, key=lambda s: min(len(s), 180))[:180] or None


def _parse_expiry_candidate(value: str) -> str | None:
    value = _normalize_digits(value)
    value = re.sub(DIRECTIONAL_MARKS, "", value)
    candidates = []
    for m in re.finditer(r"(?<!\d)(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)", value):
        try:
            candidates.append(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            pass
    for m in re.finditer(r"(?<!\d)(\d{1,2})[-/.](\d{1,2})[-/.](20\d{2})(?!\d)", value):
        try:
            candidates.append(date(int(m.group(3)), int(m.group(2)), int(m.group(1))))
        except ValueError:
            pass
    today = date.today()
    candidates = sorted({d for d in candidates if today <= d <= date(today.year + 20, 12, 31)})
    return candidates[0].isoformat() if len(candidates) == 1 else None


def _extract_expiry_date(texts: list[str]) -> str | None:
    labels = re.compile(r"البطاقة\s*سارية\s*حتى|تاريخ\s*الانتهاء|صالحة?\s*حتى|سارية?\s*حتى|expiry|valid\s*until", re.IGNORECASE)
    date_pattern = re.compile(r"(?<!\d)(?:20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]20\d{2})(?!\d)")
    for raw in texts:
        text = re.sub(DIRECTIONAL_MARKS, "", _normalize_digits(raw))
        for label in labels.finditer(text):
            nearby = text[label.end():label.end() + 100]
            match = date_pattern.search(nearby)
            if match:
                parsed = _parse_expiry_candidate(match.group(0))
                if parsed:
                    return parsed
    return None


def _extract_expiry_with_qwen(back_path: str) -> str | None:
    from PIL import Image
    from utils.ollama_client import ollama_generate

    with Image.open(back_path) as source:
        image = source.convert("RGB")
        image.thumbnail((1400, 1000))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=94)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    prompt = (
        "Inspect the back side of this Egyptian national ID. Read only the expiry date "
        "immediately following the Arabic phrase 'البطاقة سارية حتى'. "
        'Return JSON only: {"expiry_date":"YYYY-MM-DD"} or {"expiry_date":null}. '
        "Do not return date of birth or issue date. Never guess; use null if unreadable."
    )
    payload = {
        "model": os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b").strip(),
        "prompt": prompt,
        "images": [encoded],
        "stream": False,
        "keep_alive": "2m",
        "format": "json",
        "options": {"temperature": 0, "num_predict": 100, "num_ctx": 4096},
    }
    try:
        body = ollama_generate(payload, timeout=180)
        raw = str(body.get("response") or "").strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
            candidate = parsed.get("expiry_date") if isinstance(parsed, dict) else raw
        except json.JSONDecodeError:
            candidate = raw
        if not candidate or str(candidate).strip().lower() in {"null", "none", "unknown"}:
            return None
        result = _parse_expiry_candidate(str(candidate))
        if result:
            return result
        try:
            d = date.fromisoformat(str(candidate).strip())
            if date.today() <= d <= date(date.today().year + 20, 12, 31):
                return d.isoformat()
        except ValueError:
            pass
        print("    [national_id] Qwen expiry response was not a valid future date")
        return None
    except Exception as exc:
        print(f"    [national_id] Qwen expiry extraction failed: {type(exc).__name__}: {exc}")
        return None


def _extract_via_repo(front_path: str, back_path: str) -> dict:
    repo_path = Path(NATIONAL_ID_OCR_PATH)

    os.chdir(str(repo_path))
    
    if not NATIONAL_ID_OCR_PATH or not repo_path.is_dir():
        raise FileNotFoundError(f"Repo path not found: {NATIONAL_ID_OCR_PATH}")
    print(f"    [DEBUG] CWD: {os.getcwd()}")
    print(f"    [DEBUG] Files here: {os.listdir('.')}")
    repo_str = str(repo_path.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    import id_card_data_extractor as mod
    from card_recognition import scan_id_card

    module_fields = (
        "extracted_full_name", "extracted_address", "extracted_id_number", "extracted_birth_date",
        "extracted_governorate", "extracted_job_title", "extracted_workplace", "extracted_gender",
        "extracted_religion", "extracted_marital_status", "extracted_spouse_name",
    )
    output_stream = io.StringIO()
    try:
        with contextlib.redirect_stdout(output_stream):
            for field in module_fields:
                setattr(mod, field, "")
            scan_id_card(front_path, "front")
            scan_id_card(back_path, "back")
            mod.extract_front_side_data("temp_front.jpg")
            mod.extract_back_side_data("temp_back.jpg")
    except Exception as repo_inner_exc:
        
        print(f"    [DEBUG] Repo inner error: {type(repo_inner_exc).__name__}: {repo_inner_exc}")
        print(f"    [DEBUG] Repo stdout so far: {output_stream.getvalue()[:300]}")
        raise

    print(f"    [DEBUG] Repo stdout: {output_stream.getvalue()[:500]}")
    print(f"    [DEBUG] Extracted ID: {repr(getattr(mod, 'extracted_id_number', 'NOT FOUND'))}")
    repo_output = output_stream.getvalue()

    data = {
        "full_name": getattr(mod, "extracted_full_name", None),
        "national_id": getattr(mod, "extracted_id_number", None),
        "date_of_birth": getattr(mod, "extracted_birth_date", None),
        "governorate": getattr(mod, "extracted_governorate", None),
        "address": getattr(mod, "extracted_address", None),
        "profession_on_id": getattr(mod, "extracted_job_title", None),
        "gender": getattr(mod, "extracted_gender", None),
        "marital_status_on_id": getattr(mod, "extracted_marital_status", None),
        "id_expiry_date": None,
    }
    if not _is_missing(data.get("national_id")):
        repo_nid = _find_national_id(str(data["national_id"]))
        data["national_id"] = repo_nid or data["national_id"]

    if _is_missing(data.get("national_id")):
        data["national_id"] = _find_national_id(repo_output)

    for key, value in list(data.items()):
        if isinstance(value, str):
            value = value.strip()
            if value.lower() in {"", "unknown", "none", "null", "غير محدد"}:
                value = None
            data[key] = value
    if data.get("date_of_birth"):
        data["date_of_birth"] = str(data["date_of_birth"]).replace("/", "-")
    if _is_missing(data.get("national_id")):
        print("    [national_id] Repo ID extraction failed; other fields may still be available")
        print("    [national_id] Repo field availability:", {k: not _is_missing(v) for k, v in data.items()})
        raise ValueError("Repo could not extract the national ID number")
    return data


def _extract_name_address_from_image(front_path: str) -> dict:
    import cv2
    import pytesseract
    image = cv2.imread(str(front_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        image = _to_gray_image(_read_pages(front_path)[0])
    image = _deskew_and_orient(image)
    image = cv2.resize(image, (1000, 630), interpolation=cv2.INTER_CUBIC)
    results = {"full_name": None, "address": None}

    for field, bounds in (("full_name", NAME_CROP), ("address", ADDRESS_CROP)):
        y1, y2, x1, x2 = bounds
        if not (0 <= y1 < y2 <= 630 and 0 <= x1 < x2 <= 1000):
            raise ValueError(f"Invalid {field} crop: {bounds}")
        region = image[y1:y2, x1:x2]
        enlarged = cv2.resize(region, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        if SAVE_DEBUG_CROPS:
            cv2.imwrite(f"debug_{field}_crop.png", enlarged)
        raw_results = []
        for psm in (11, 6, 7):
            try:
                raw_results.append(pytesseract.image_to_string(enlarged, lang="ara", config=f"--oem 1 --psm {psm}") or "")
            except Exception as exc:
                print(f"    [national_id] {field} PSM {psm} failed: {type(exc).__name__}")
        lines, seen = [], set()
        for raw in raw_results:
            for line in raw.splitlines():
                cleaned = _clean_text(line)
                if cleaned and cleaned not in seen:
                    seen.add(cleaned)
                    lines.append(cleaned)

        if field == "full_name":
            candidates = []
            for line in lines:
                if re.search(r"\d", line):
                    continue
                words = re.findall(r"[\u0621-\u064a]+", line)
                if 2 <= len(words) <= 5:
                    candidate = " ".join(words)
                    if len(candidate) >= 5:
                        candidates.append(candidate)
            if candidates:
                results[field] = max(candidates, key=lambda s: (len(s.split()) in (3, 4), len(s)))
        else:
            candidates = [line for line in lines if len(line) >= 3 and len(re.findall(r"[\u0621-\u064a]", line)) >= 2]
            candidate = _clean_text(" ".join(candidates[:5]))
            candidate = re.sub(r'^([0-9]{1,4})?\s*["\'""`]+\s*', lambda m: ((m.group(1) or "") + " "), candidate).strip()
            if len(candidate) >= 10 and len(re.findall(r"[\u0621-\u064a]", candidate)) >= 6:
                results[field] = candidate
    return results


def _extract_via_tesseract(front_path: str, back_path: str) -> dict:
    import cv2
    import pytesseract

    front_page = _read_pages(front_path)[0]
    back_page = _read_pages(back_path)[0]

    # تصحيح الـ rotation قبل الـ OCR
    front_gray = _deskew_and_orient(_to_gray_image(front_page))
    back_gray = _deskew_and_orient(_to_gray_image(back_page))

    # تكبير الصورة لو صغيرة
    import cv2 as _cv2
    for attr_name in ("front_gray", "back_gray"):
        img = locals()[attr_name]
        if max(img.shape[:2]) < 1800:
            locals()[attr_name]  # just reference

    if max(front_gray.shape[:2]) < 1800:
        front_gray = cv2.resize(front_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if max(back_gray.shape[:2]) < 1800:
        back_gray = cv2.resize(back_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    front_text = _normalize_digits(
        pytesseract.image_to_string(front_gray, lang="ara+eng", config="--oem 1 --psm 6") or ""
    )
    back_text = _normalize_digits(
        pytesseract.image_to_string(back_gray, lang="ara+eng", config="--oem 1 --psm 6") or ""
    )

    print(f"    [national_id] Front OCR sample: {repr(front_text[:150])}")

    back_variants = _ocr_back_variants(back_path)
    all_back_text = [back_text] + back_variants
    full_text = "\n".join([front_text, *all_back_text])

    nid = _find_national_id([front_text, *all_back_text])
    print(f"    [national_id] Tesseract ID candidate found: {bool(nid)}")

    name = _extract_labeled_field(front_text, [r"الاسم\s+رباعي(?:اً|ا)?", r"الاسم\s+بالكامل", r"الاسم", r"\bName\b"], 5, 70)
    address = _extract_labeled_field(front_text, [r"عنوان\s+الإقامة", r"عنوان\s+الاقامة", r"محل\s+الإقامة", r"محل\s+الاقامة", r"العنوان", r"\bAddress\b"], 5, 150)
    profession = _extract_profession(all_back_text)
    expiry = _extract_expiry_date(all_back_text)
    marital = None
    for pattern, status in ((r"أعزب|عزباء", "أعزب"), (r"متزوج|متزوجة", "متزوج"), (r"مطلق|مطلقة", "مطلق"), (r"أرمل|أرملة", "أرمل")):
        if re.search(pattern, full_text):
            marital = status
            break
    result = {
        "full_name": name, "national_id": nid, "address": address,
        "id_expiry_date": expiry, "marital_status_on_id": marital,
        "profession_on_id": profession,
    }
    if nid:
        result.update(_parse_id_number(nid))
    return result


def _extract_via_ollama_vision(front_path: str, back_path: str) -> dict:
    from utils.ollama_client import ollama_generate
    def encode(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    prompt = (
        "Extract fields from the front and back of an Egyptian national ID. Return only JSON with keys: "
        "full_name, national_id, date_of_birth, governorate, address, gender, marital_status_on_id, "
        "profession_on_id, id_expiry_date. Use Arabic for Arabic text; dates YYYY-MM-DD; null if unreadable. "
        "Never guess. ID must be exactly 14 digits. Expiry is the date printed after 'البطاقة سارية حتى', "
        "not birth or issue date."
    )
    payload = {
        "model": os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b").strip(),
        "prompt": prompt, "images": [encode(front_path), encode(back_path)],
        "stream": False, "format": "json", "keep_alive": "2m",
        "options": {"temperature": 0, "num_predict": 700, "num_ctx": 4096},
    }
    body = ollama_generate(payload, timeout=300)
    raw = str(body.get("response") or "").strip()
    raw = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw, flags=re.IGNORECASE).strip()
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Vision model did not return a JSON object")
    print("    [national_id] Qwen vision extraction succeeded")
    return parsed


def extract(front_path: str, back_path: str) -> dict:
    front_path, back_path = str(front_path), str(back_path)
    repo_succeeded = False
    try:
        print("    [national_id] Trying Marwan-Ghazal OCR repo...")
        data = _extract_via_repo(front_path, back_path)
        repo_succeeded = True
        print("    [national_id] Repo succeeded")
    except Exception as exc:
        print(f"    [national_id] Repo failed ({type(exc).__name__}: {exc}); trying Tesseract")
        data = _extract_via_tesseract(front_path, back_path)
        print("    [national_id] Tesseract fallback done")

    if repo_succeeded and any(_is_missing(data.get(k)) for k in EXPECTED_KEYS):
        try:
            supplemental = _extract_via_tesseract(front_path, back_path)
            for key, value in supplemental.items():
                if _is_missing(data.get(key)) and not _is_missing(value):
                    data[key] = value
            print("    [national_id] Tesseract supplemental recovery completed")
        except Exception as exc:
            print(f"    [national_id] Supplemental OCR failed: {type(exc).__name__}: {exc}")

    for key in EXPECTED_KEYS:
        data.setdefault(key, None)

    try:
        refined = _extract_name_address_from_image(front_path)
        for field in ("full_name", "address"):
            value = refined.get(field)
            current = data.get(field)
            if value and _is_missing(current):
                data[field] = value
        print("    [national_id] Name/address crop refinement completed")
    except Exception as exc:
        print(f"    [national_id] Name/address refinement failed: {type(exc).__name__}: {exc}")

    back_texts = []
    try:
        back_texts = _ocr_back_variants(back_path)
        if _is_missing(data.get("id_expiry_date")):
            expiry = _extract_expiry_date(back_texts)
            if expiry:
                data["id_expiry_date"] = expiry
                print("    [national_id] Expiry date recovered by OCR")
        profession = _extract_profession(back_texts)
        current_profession = data.get("profession_on_id")
        if profession and (_is_missing(current_profession) or len(profession) > len(str(current_profession)) + 8):
            data["profession_on_id"] = profession
            print("    [national_id] Profession recovered/refined")
    except Exception as exc:
        print(f"    [national_id] Back-side recovery failed: {type(exc).__name__}: {exc}")

    if _is_missing(data.get("id_expiry_date")):
        expiry = _extract_expiry_with_qwen(back_path)
        if expiry:
            data["id_expiry_date"] = expiry
            print("    [national_id] Expiry date recovered by Qwen")
        else:
            data["id_expiry_date"] = None
            print("    [national_id] Expiry date remains unreadable")

    placeholders = {"arabic full name", "full name", "yyyy-mm-dd", "governorate in arabic", "full address", "profession text", "name", "address", "profession", "marital status", "unknown", "غير محدد"}
    def missing(value):
        return _is_missing(value) or (isinstance(value, str) and value.strip().lower() in placeholders)
    vision_fields = ("full_name", "national_id", "address", "profession_on_id", "id_expiry_date")
    if os.getenv("NATIONAL_ID_USE_LEGACY_VISION", "0") == "1" and any(missing(data.get(k)) for k in vision_fields):
        try:
            vision_data = _extract_via_ollama_vision(front_path, back_path)
            for key in EXPECTED_KEYS:
                value = vision_data.get(key)
                if not missing(value) and missing(data.get(key)):
                    data[key] = value
        except Exception as exc:
            print(f"    [national_id] Ollama vision unavailable: {type(exc).__name__}: {exc}")

    nid_value = next((data.get(k) for k in ("national_id", "id_number", "ID Number", "id", "National ID", "national_id_number") if not _is_missing(data.get(k))), None)
    if nid_value is not None:
        nid = re.sub(r"\D", "", _normalize_digits(nid_value))
        if re.fullmatch(r"[23]\d{13}", nid):
            data["national_id"] = nid
            decoded = _parse_id_number(nid)
            data.update(decoded)
        else:
            data["national_id"] = nid or None
            print("    [national_id] Warning: extracted ID does not match the expected 14-digit format")

    for field in ("date_of_birth", "id_expiry_date"):
        value = data.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        value = value.strip()
        try:
            data[field] = date.fromisoformat(value.replace("/", "-")).isoformat()
        except ValueError:
            parsed = _parse_expiry_candidate(value) if field == "id_expiry_date" else None
            data[field] = parsed or value

    result = {key: data.get(key) for key in EXPECTED_KEYS}
    if "age_years" in data:
        result["age_years"] = data["age_years"]
    return result


def extract_to_json(front_path: str, back_path: str, output_path: str = "national_id_result.json") -> dict:
    data = extract(front_path, back_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    [national_id] JSON saved to: {output.resolve()}")
    return data

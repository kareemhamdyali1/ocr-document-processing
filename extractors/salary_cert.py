"""
Salary Certificate extractor.

Primary:  Tesseract OCR + regex rules
Fallback: Ollama local LLM (parses the OCR text, no image sent)
"""

import re
import json
from datetime import date, datetime
from PIL import Image

from utils.pdf_to_images import load_document
from utils.ocr import ocr_pages


SCHEMA = {
    "employer_name": None,
    "employer_sector": None,
    "job_title": None,
    "declared_net_salary": None,
    "declared_gross_salary": None,
    "employment_date": None,
    "employment_tenure_years": None,
    "issue_date": None,
}

# ── regex rules ───────────────────────────────────────────────────────────────

def _parse_arabic_date(text: str) -> str | None:
    """Convert DD/MM/YYYY or YYYY/MM/DD or YYYY-MM-DD to ISO YYYY-MM-DD."""
    m = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", text)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", text)
    if m:
        return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    return None


def _tenure(employment_date_iso: str | None) -> float | None:
    if not employment_date_iso:
        return None
    try:
        emp = datetime.strptime(employment_date_iso, "%Y-%m-%d").date()
        return round((date.today() - emp).days / 365.25, 1)
    except ValueError:
        return None


def _money(text: str) -> float | None:
    """Extract first number that looks like a salary."""
    m = re.search(r"[\d,،]+(?:\.\d+)?", text)
    if m:
        return float(m.group().replace(",", "").replace("،", ""))
    return None


MONTH_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}

def _parse_english_date(text: str) -> str | None:
    """Parse dates like '05-Oct-2024', '1-Apr-2017', '10-Dec-2023'."""
    m = re.search(r"(\d{1,2})[-/]([A-Za-z]{3})[-/](\d{4})", text)
    if m:
        month = MONTH_MAP.get(m.group(2).lower())
        year = int(m.group(3))
        # Reject clearly wrong future years (OCR sometimes reads 2017 as 2037)
        current_year = date.today().year
        if year > current_year:
            year = year - 20  # e.g. 2037 → 2017
        if month:
            return f"{year}-{month}-{int(m.group(1)):02d}"
    return _parse_arabic_date(text)


def _extract_via_regex(text: str) -> dict:
    result = dict(SCHEMA)

    # Employer name — prefer CURRENT employer ("currently employed by X from")
    # The cert has two employers: old one and current one
    m = re.search(
        r"currently employed by\s+(.+?)\s+from\s+\d",
        text, re.IGNORECASE
    )
    if m:
        result["employer_name"] = m.group(1).strip()
    else:
        # Find all "employed by X" occurrences and take the LAST one (most recent)
        matches = list(re.finditer(r"employed by\s+(.+?)\s+(?:for|from|as)", text, re.IGNORECASE))
        if matches:
            result["employer_name"] = matches[-1].group(1).strip()
        else:
            for pattern in [
                r"(?:جهة العمل|اسم الشركة|المنشأة|صاحب العمل)[:\s]*([^\n]{3,80})",
            ]:
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    result["employer_name"] = m.group(1).strip()
                    break

    # Job title — "as Senior Care Ops Team Lead from" — take the LAST occurrence
    matches = list(re.finditer(r"\bas\s+(.+?)\s+from\s+\d", text, re.IGNORECASE))
    if matches:
        result["job_title"] = matches[-1].group(1).strip()
    else:
        for pattern in [
            r"(?:المسمى الوظيفي|الوظيفة|المهنة)[:\s]*([^\n]{3,60})",
            r"(?:Job Title|Position)[:\s]*([^\n]{3,60})",
        ]:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                result["job_title"] = m.group(1).strip()
                break

    # Gross salary — "monthly gross salary of 28,600 EGP"
    m = re.search(
        r"(?:monthly gross salary of|gross salary)[^\d]*([\d,،]+(?:\.\d+)?)",
        text, re.IGNORECASE
    )
    if m:
        raw = m.group(1).replace(",", "").replace("،", "")
        try:
            result["declared_gross_salary"] = float(raw)
        except ValueError:
            pass
    else:
        for pattern in [
            r"(?:الراتب الإجمالي|الأجر الإجمالي|المرتب الإجمالي|Gross)[^\n]*?([\d,،]+(?:\.\d+)?)",
        ]:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                raw = m.group(1).replace(",", "").replace("،", "")
                try:
                    result["declared_gross_salary"] = float(raw)
                except ValueError:
                    pass
                break

    # Net salary — only set if explicitly mentioned and value > 100 (avoid OCR noise)
    for pattern in [
        r"(?:monthly net salary of|net salary)[^\d]*([\d,،]+(?:\.\d+)?)",
        r"(?:الراتب الصافي|الأجر الصافي|المرتب الصافي)[^\n]*?([\d,،]+(?:\.\d+)?)",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            raw = m.group(1).replace(",", "").replace("،", "")
            try:
                val = float(raw)
                if val > 100:   # ignore OCR noise like "1.0"
                    result["declared_net_salary"] = val
            except ValueError:
                pass
            break
        else:
            result["declared_net_salary"] = result["declared_gross_salary"] * 0.8 

    # Employment date — current job start: "currently employed by X from DD-Mon-YYYY"
    m = re.search(
        r"currently employed by.+?from\s+(\d{1,2}[-/][A-Za-z]{3}[-/]\d{4})",
        text, re.IGNORECASE | re.DOTALL
    )
    if m:
        result["employment_date"] = _parse_english_date(m.group(1))
    else:
        # Fallback: any "from DD-Mon-YYYY" pattern
        dates = re.findall(r"from\s+(\d{1,2}[-/][A-Za-z]{3}[-/]\d{4})", text, re.IGNORECASE)
        if dates:
            result["employment_date"] = _parse_english_date(dates[-1])

    # Issue date — "Date: 05-Oct-2024" — must be a full parseable date
    m = re.search(r"Date[:\s]+(\d{1,2}[-/][A-Za-z]{3}[-/]\d{4})", text, re.IGNORECASE)
    if m:
        parsed = _parse_english_date(m.group(1))
        if parsed:
            result["issue_date"] = parsed
    if not result["issue_date"]:
        for pattern in [
            r"(?:تاريخ الإصدار|تاريخ الشهادة|صادر في)[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{4})",
        ]:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                result["issue_date"] = _parse_arabic_date(m.group(1))
                break

    # Sector heuristic
    name_lower = (result.get("employer_name") or "").lower()
    if any(w in name_lower for w in ["llc", "inc", "corp", "ltd", "plc", "group"]):
        result["employer_sector"] = "Multinational Corporation"
    elif any(w in name_lower for w in ["bank", "بنك"]):
        result["employer_sector"] = "Banking"
    elif any(w in name_lower for w in ["حكومة", "وزارة", "هيئة"]):
        result["employer_sector"] = "Government"

    result["employment_tenure_years"] = _tenure(result["employment_date"])
    return result


# ── Ollama fallback ───────────────────────────────────────────────────────────

def _extract_via_ollama(text: str, model: str) -> dict:
    from utils.ollama_client import ollama_extract

    prompt = f"""You are a data extraction assistant.
Extract all salary certificate fields from the following OCR text and return ONLY a valid JSON object.
Use null for any field you cannot find. Dates must be in YYYY-MM-DD format. Salaries as numbers (no commas).

Required JSON schema:
{json.dumps(SCHEMA, ensure_ascii=False, indent=2)}

OCR TEXT:
\"\"\"
{text[:4000]}
\"\"\"

Return ONLY the JSON object, no explanation."""

    return ollama_extract(prompt, model=model)


# ── public entry point ────────────────────────────────────────────────────────

def extract(path: str, ollama_model: str = "qwen2.5:7b") -> dict:
    pages = load_document(path)
    text  = ocr_pages(pages, lang="ara+eng")

    print("    [salary_cert] Running Tesseract regex extraction…")
    result = _extract_via_regex(text)

    # Check if key fields are missing — if so, try Ollama
    missing = [k for k in ("employer_name", "job_title", "declared_gross_salary")
               if not result.get(k)]

    if missing:
        print(f"    [salary_cert] Regex missed: {missing} — trying Ollama…")
        try:
            from utils.ollama_client import is_ollama_running
            if not is_ollama_running():
                raise RuntimeError("Ollama not running")
            ollama_result = _extract_via_ollama(text, model=ollama_model)
            # Merge: Ollama fills only the gaps
            for k, v in ollama_result.items():
                if v is not None and result.get(k) is None:
                    result[k] = v
            # Recalculate tenure if employment_date was filled by Ollama
            result["employment_tenure_years"] = _tenure(result.get("employment_date"))
            print("    [salary_cert] ✓ Ollama filled gaps")
        except Exception as e:
            print(f"    [salary_cert] Ollama unavailable ({e}) — using regex result as-is")
    else:
        print("    [salary_cert] ✓ Regex extraction complete")

    return result

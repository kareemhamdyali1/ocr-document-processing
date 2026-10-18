import re
import unicodedata
from difflib import SequenceMatcher


# Normalize Arabic and Persian digits to ASCII.
_AR_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)

# Labels used to identify national ID numbers in OCR text.
_LABEL = re.compile(
    r"(?:"
    r"national\s*id|national\s*identity|identity\s*(?:no|number)|"
    r"id\s*(?:no|number)|"
    r"الرقم\s*القومي|الرقم\s*القومى|رقم\s*قومي|رقم\s*قومى|"
    r"رقم\s*البطاقة|رقم\s*الهوية"
    r")",
    re.IGNORECASE,
)

_ID = re.compile(r"(?<!\d)\d{14}(?!\d)")

# Labels used to identify names in structured fields and OCR text.
_NAME_KEYS = (
    "full_name",
    "name",
    "name_ar",
    "customer_name",
    "account_holder",
    "account_holder_name",
    "client_name",
    "applicant_name",
    "employee_name",
    "holder_name",
)

_NAME_LABEL = re.compile(
    r"(?:"
    r"applicant\s*name|customer\s*name|client\s*name|"
    r"account\s*holder(?:\s*name)?|holder\s*name|"
    r"employee\s*name|name\s*of\s*employee|full\s*name|"
    r"اسم\s*(?:الموظف|العامل|مقدم\s*الطلب|المتقدم|العميل|"
    r"صاحب\s*الحساب|صاحب\s*التقرير|المستعلم)|"
    r"الاسم\s*(?:الكامل)?"
    r")",
    re.IGNORECASE,
)

_NAME_LABEL_LINE = re.compile(
    r"^\s*(?:"
    r"applicant\s*name|customer\s*name|client\s*name|"
    r"account\s*holder(?:\s*name)?|holder\s*name|"
    r"employee\s*name|name\s*of\s*employee|full\s*name|"
    r"اسم\s*(?:الموظف|العامل|مقدم\s*الطلب|المتقدم|العميل|"
    r"صاحب\s*الحساب|صاحب\s*التقرير|المستعلم)|"
    r"الاسم\s*(?:الكامل)?"
    r")\s*[:：\-]\s*(.{3,100}?)\s*$",
    re.IGNORECASE,
)


def _value(data, key):
    """Read a field whether it is a plain value or {'value': ...}."""
    if not isinstance(data, dict):
        return None

    value = data.get(key)

    if isinstance(value, dict):
        value = value.get("value")

    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()

    return value if value != "" else None


def _normalize_id(value):
    """Normalize an ID candidate and accept only exactly 14 digits."""
    if value is None:
        return None

    value = str(value).translate(_AR_DIGITS)
    digits = re.sub(r"\D", "", value)

    if len(digits) != 14:
        return None

    return digits


def _normalize_name(value):
    """Normalize Arabic/English names before fuzzy comparison."""
    if value is None:
        return ""

    value = unicodedata.normalize("NFKC", str(value))
    value = value.translate(_AR_DIGITS).lower()

    # Remove Arabic diacritics and tatweel.
    value = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", value)

    # Normalize common Arabic letter variants.
    value = value.translate(
        str.maketrans({
            "أ": "ا",
            "إ": "ا",
            "آ": "ا",
            "ٱ": "ا",
            "ى": "ي",
            "ؤ": "و",
            "ئ": "ي",
        })
    )

    # Keep Arabic/English letters and digits; remove punctuation.
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def _similarity(name_a, name_b):
    """Return a similarity score between 0 and 1."""
    a = _normalize_name(name_a)
    b = _normalize_name(name_b)

    if not a or not b:
        return 0.0

    sequence_score = SequenceMatcher(None, a, b).ratio()

    tokens_a = set(a.split())
    tokens_b = set(b.split())

    if tokens_a or tokens_b:
        token_score = (
            2 * len(tokens_a & tokens_b)
            / (len(tokens_a) + len(tokens_b))
        )
    else:
        token_score = 0.0

    return max(sequence_score, token_score)


def _raw_text(data):
    """Read OCR text from common extractor output keys."""
    if not isinstance(data, dict):
        return ""

    for key in (
        "raw_text",
        "ocr_text",
        "full_text",
        "extracted_text",
        "text",
    ):
        value = data.get(key)

        if isinstance(value, str) and value.strip():
            return value.translate(_AR_DIGITS)

    return ""


def _doc_fields(payload, document_type):
    """Find extracted fields for a document in the assembled payload."""
    documents = payload.get("documents", [])

    if not isinstance(documents, list):
        return {}

    for document in documents:
        if not isinstance(document, dict):
            continue

        current_type = document.get("document_type")

        if current_type == document_type:
            fields = document.get("extracted_fields", {})
            return fields if isinstance(fields, dict) else {}

    return {}


def _ids_from_text(raw_text):
    """
    Extract 14-digit IDs only when they appear near a national-ID label.
    This avoids treating account numbers or unrelated 14-digit strings
    as national IDs.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return []

    text = raw_text.translate(_AR_DIGITS)
    candidates = []

    for match in _LABEL.finditer(text):
        nearby = text[match.end():match.end() + 100]

        # Prefer the number following the label.
        id_match = _ID.search(nearby)

        if id_match:
            candidates.append(id_match.group(0))

    return list(dict.fromkeys(candidates))


def _name_from_text(raw_text):
    """Extract a name only when OCR contains an explicit name label."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None

    text = raw_text.translate(_AR_DIGITS)

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        match = _NAME_LABEL_LINE.match(line)

        if not match:
            continue

        candidate = re.sub(r"\s+", " ", match.group(1)).strip(" \t:：-")

        if (
            len(candidate) >= 3
            and re.search(r"[A-Za-z\u0621-\u064A]", candidate)
            and not re.fullmatch(r"[\d\s./-]+", candidate)
        ):
            return candidate

    return None


def extract_identity_hints(raw_text, kind):
    """
    Extract only the identity hints needed for cross-document checks.

    Store these hints inside the extractor's internal result. The final
    response should expose only consistency_checks, not this metadata.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return {"national_ids": [], "names": []}

    text = raw_text.translate(_AR_DIGITS)
    national_ids = _ids_from_text(text)

    labels_by_kind = {
        "salary_certificate": True,
        "bank_statement": True,
        "iscore_report": True,
    }

    names = []

    if labels_by_kind.get(kind, False):
        name = _name_from_text(text)

        if name:
            names.append(name)

    return {
        "national_ids": list(dict.fromkeys(national_ids)),
        "names": list(dict.fromkeys(names)),
    }


def _get_field(data, keys):
    """Return the first non-empty value from a sequence of field names."""
    if not isinstance(data, dict):
        return None

    for key in keys:
        value = _value(data, key)

        if value is not None:
            return value

    return None


def _get_source(source_results, kind):
    if not isinstance(source_results, dict):
        return {}

    source = source_results.get(kind, {})
    return source if isinstance(source, dict) else {}


def _get_identity_hint(source):
    hint = source.get("_consistency_identity", {})

    return hint if isinstance(hint, dict) else {}


def _collect_ids(fields, source):
    candidates = []

    # Structured fields.
    for key in (
        "national_id",
        "national_id_number",
        "id_number",
        "identity_number",
        "national_number",
    ):
        value = _get_field(fields, (key,))

        normalized = _normalize_id(value)

        if normalized:
            candidates.append(normalized)

    # OCR text, but only IDs near an explicit national-ID label.
    candidates.extend(_ids_from_text(_raw_text(source)))

    # Private identity hints extracted from OCR by the extractor.
    hint = _get_identity_hint(source)

    for candidate in hint.get("national_ids", []):
        normalized = _normalize_id(candidate)

        if normalized:
            candidates.append(normalized)

    return list(dict.fromkeys(candidates))


def _collect_name(fields, source, keys):
    name = _get_field(fields, keys)

    if isinstance(name, str) and name.strip():
        return name.strip()

    # Try explicit name labels in raw OCR text.
    name = _name_from_text(_raw_text(source))

    if name:
        return name

    # Fall back to internal identity hints.
    hint = _get_identity_hint(source)
    hinted_names = hint.get("names", [])

    if isinstance(hinted_names, list):
        for candidate in hinted_names:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

    return None


def _quality_score(document):
    """Return a quality score when one is present; otherwise None."""
    if not isinstance(document, dict):
        return None

    for key in (
        "overall_quality_score",
        "quality_score",
        "confidence_score",
    ):
        value = document.get(key)

        if isinstance(value, dict):
            value = value.get("value")

        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue

    return None


def build_consistency_checks(payload, source_results=None):
    """
    Compare identity data across the national ID, bank statement,
    salary certificate, and iScore report.

    Expected payload keys:
      national_id_fields
      salary_certificate_fields
      bank_statement_fields
      iscore_report_fields
      documents

    source_results may contain the original extractor outputs. It is used
    for internal OCR hints and is not copied into the returned result.
    """
    if not isinstance(payload, dict):
        payload = {}

    if not isinstance(source_results, dict):
        source_results = {}

    warnings = []

    nid_fields = payload.get("national_id_fields", {})
    if not isinstance(nid_fields, dict):
        nid_fields = {}

    expected_id = _normalize_id(
        _get_field(nid_fields, ("national_id", "id_number"))
    )

    expected_name = _get_field(
        nid_fields,
        ("full_name", "name_ar", "name"),
    )

    # Fall back to the national ID's document summary if needed.
    if not expected_id:
        expected_id = _normalize_id(
            _get_field(
                _doc_fields(payload, "national_id"),
                ("national_id", "id_number"),
            )
        )

    if not expected_name:
        expected_name = _get_field(
            _doc_fields(payload, "national_id"),
            _NAME_KEYS,
        )

    if not expected_id:
        warnings.append(
            "National ID consistency could not be checked: "
            "national ID number is missing or invalid."
        )

    if not expected_name:
        warnings.append(
            "Name consistency could not be checked: "
            "name on the national ID is missing."
        )

    # Check ID number against bank statement and iScore.
    id_checks = []

    for kind, label in (
        ("bank_statement", "bank statement"),
        ("iscore_report", "iScore report"),
    ):
        fields = payload.get(f"{kind}_fields", {})
        if not isinstance(fields, dict):
            fields = {}

        if not fields:
            fields = _doc_fields(payload, kind)

        source = _get_source(source_results, kind)
        candidates = _collect_ids(fields, source)

        if not candidates:
            warnings.append(
                f"National ID consistency could not be verified for "
                f"{label}: no labeled or structured national ID was found."
            )
            id_checks.append(False)
            continue

        if not expected_id:
            id_checks.append(False)
            continue

        matched = expected_id in candidates
        id_checks.append(matched)

        if not matched:
            warnings.append(
                f"National ID mismatch: the national ID on the "
                f"{label} does not match the national ID card."
            )

    # Name comparison against salary certificate and bank statement.
    name_checks = []

    for kind, label, keys in (
        (
            "salary_certificate",
            "salary certificate",
            ("employee_name", "full_name", "name", "name_ar"),
        ),
        (
            "bank_statement",
            "bank statement",
            (
                "customer_name",
                "account_holder",
                "account_holder_name",
                "client_name",
                "full_name",
                "name",
                "name_ar",
            ),
        ),
    ):
        fields = payload.get(f"{kind}_fields", {})
        if not isinstance(fields, dict):
            fields = {}

        if not fields:
            fields = _doc_fields(payload, kind)

        source = _get_source(source_results, kind)
        candidate_name = _collect_name(fields, source, keys)

        if not candidate_name:
            warnings.append(
                f"Name consistency could not be verified for "
                f"{label}: no usable name was found."
            )
            name_checks.append(False)
            continue

        if not expected_name:
            name_checks.append(False)
            continue

        score = _similarity(expected_name, candidate_name)

        if score < 0.80:
            warnings.append(
                f"Name mismatch: the name on the {label} differs "
                f"from the name on the national ID card "
                f"(similarity: {score:.0%})."
            )
            name_checks.append(False)
        else:
            name_checks.append(True)

    # Flag low-quality documents when a quality score is available.
    documents = payload.get("documents", [])

    if isinstance(documents, list):
        for document in documents:
            if not isinstance(document, dict):
                continue

            quality = _quality_score(document)

            if quality is not None and quality < 0.50:
                document_type = document.get(
                    "document_type",
                    "unknown document",
                )
                warnings.append(
                    f"Low document quality: {document_type} "
                    f"has a quality score of {quality:.0%}."
                )

    # Warn if the OCR output contains Unicode replacement characters.
    for kind, source in source_results.items():
        if not isinstance(source, dict):
            continue

        raw = _raw_text(source)

        if "\ufffd" in raw:
            warnings.append(
                f"OCR text for {kind} contains invalid characters; "
                "review the extracted data."
            )

    # Preserve order while removing duplicate warnings.
    warnings = list(dict.fromkeys(warnings))

    return {
        "national_id_consistent": bool(
            expected_id and id_checks and all(id_checks)
        ),
        "name_consistent": bool(
            expected_name and name_checks and all(name_checks)
        ),
        "warnings": warnings,
    }

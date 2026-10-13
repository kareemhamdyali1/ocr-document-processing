"""Evidence-based validation scores (not calibrated OCR probabilities)."""
from datetime import date, datetime
import re


def _valid_date(value):
    if not isinstance(value, str):
        return False
    try:
        return datetime.strptime(value, '%Y-%m-%d').date().isoformat() == value
    except ValueError:
        return False


def _egyptian_id_valid(value):
    if not isinstance(value, str) or not re.fullmatch(r'[23]\d{13}', value):
        return False
    century = 1900 if value[0] == '2' else 2000
    try:
        born = date(century + int(value[1:3]), int(value[3:5]), int(value[5:7]))
        return born <= date.today()
    except ValueError:
        return False


def _score(field, value, section, result):
    if value is None or value == '' or value == []:
        return 0.0
    if field == 'national_id':
        return 0.9 if _egyptian_id_valid(value) else 0.2
    if field in {'date_of_birth', 'employment_date', 'issue_date', 'score_date', 'id_expiry_date'}:
        return 0.85 if _valid_date(value) else 0.2
    if field == 'age_years':
        return 0.85 if isinstance(value, int) and 0 <= value <= 120 else 0.25
    if field == 'gender':
        return 0.75
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return 0.8  # numeric value passed assembler validation; source unverified
    if isinstance(value, str):
        return 0.7 if len(value.strip()) >= 3 else 0.5
    return 0.6


def apply_validation_scores(result):
    """Update existing confidence numbers without changing the JSON structure.

    These scores represent basic validation evidence, NOT OCR model confidence.
    """
    for section in ('national_id_fields', 'salary_certificate_fields', 'bank_statement_fields', 'iscore_report_fields'):
        fields = result.get(section, {})
        for name, wrapped in fields.items():
            if isinstance(wrapped, dict) and 'value' in wrapped and 'confidence' in wrapped:
                wrapped['confidence'] = _score(name, wrapped['value'], section, result)
    return result

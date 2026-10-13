"""Conservative Egyptian iScore OCR extractor with optional Ollama fallback."""
import json
import re
from datetime import date
from pathlib import Path
from utils.pdf_to_images import load_document
from utils.ocr import ocr_pages

SCHEMA = {
    'credit_score': None, 'score_tier': '300 - 850', 'score_date': None,
    'total_active_loans_limit': None, 'total_outstanding_balance': None,
    'total_overdue_amount': None, 'max_days_past_due': None,
    'active_credit_cards_count': None, 'total_credit_card_utilization': None,
    'bureau_facilities': [],
}
FACILITY_SCHEMA = {
    'lender_name': None, 'facility_type': None, 'granted_amount': None,
    'outstanding_amount': None, 'installment_amount': None, 'status': None,
    'overdue_days': None, 'overdue_amount': None, 'start_date': None,
    'legal_action_flag': False,
}
MONTHS = {'يناير': 1, 'فبراير': 2, 'مارس': 3, 'ابريل': 4, 'أبريل': 4,
          'مايو': 5, 'يونيو': 6, 'يوليو': 7, 'اغسطس': 8, 'أغسطس': 8,
          'سبتمبر': 9, 'اكتوبر': 10, 'أكتوبر': 10, 'نوفمبر': 11, 'ديسمبر': 12}
DIGITS = str.maketrans('٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹', '01234567890123456789')


def _normalize(text):
    text = text.translate(DIGITS)
    text = re.sub('[\u200e\u200f\u202a-\u202e]', '', text)
    text = text.replace('ـ', '')
    return text


def _date(text):
    text = _normalize(str(text))
    for pattern, ymd in [
        (r'\b(20\d{2}|19\d{2})[/-](\d{1,2})[/-](\d{1,2})\b', True),
        (r'\b(\d{1,2})[/-](\d{1,2})[/-](20\d{2}|19\d{2})\b', False),
    ]:
        m = re.search(pattern, text)
        if m:
            a, b, c = map(int, m.groups())
            try:
                return date(a, b, c).isoformat() if ymd else date(c, b, a).isoformat()
            except ValueError:
                return None
    for name, month in MONTHS.items():
        m = re.search(r'\b(\d{1,2})\s+' + re.escape(name) + r'\s+(\d{4})\b', text)
        if m:
            try:
                return date(int(m.group(2)), month, int(m.group(1))).isoformat()
            except ValueError:
                return None
    return None


def _number(text):
    s = _normalize(str(text)).replace('٬', ',').replace('٫', '.')
    s = s.replace('،', ',').replace(',', '')
    m = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*(?:جم|جنيه|EGP)?\s*', s, re.I)
    return float(m.group(1)) if m else None


def _extract_via_regex(text):
    text = _normalize(text)
    result = {**SCHEMA, 'bureau_facilities': []}
    lines = [re.sub(r'\s+', ' ', s).strip() for s in text.splitlines() if s.strip()]

    # Score: require evidence of the 300-850 scale nearby, never just any 3-digit number.
    for i, line in enumerate(lines):
        for m in re.finditer(r'(?<!\d)([3-8]\d{2})(?!\d)', line):
            value = int(m.group(1))
            context = ' '.join(lines[max(0, i-6):i+7])
            if 300 <= value <= 850 and value not in (300, 850) and re.search(r'\b300\b', context) and re.search(r'\b850\b', context):
                result['credit_score'] = float(value)
                break
        if result['credit_score'] is not None:
            break

    for i, line in enumerate(lines):
        if re.search(r'(?:آخر|اخر|اخ|تاريخ)\s*(?:ال)?تحديث|score\s*date|report\s*date', line, re.I):
            result['score_date'] = _date(' '.join(lines[i:i+2]))
            if result['score_date']:
                break

    # Only parse amounts with an unambiguous label/value relationship.
    # In the supplied report, the summary is flattened into three labels then three
    # values; assigning 16,200 to a particular label would be speculative.
    field_labels = {
        'total_active_loans_limit': [r'إجمالي\s*الحدود\s*الائتمانية', r'total\s*credit\s*limit'],
        'total_outstanding_balance': [r'إجمالي\s*الرصيد\s*(?:القائم|المتبقي)', r'total\s*outstanding\s*balance'],
        'total_overdue_amount': [r'إجمالي\s*المتأخرات', r'total\s*overdue'],
        'max_days_past_due': [r'أقصى\s*أيام\s*التأخير', r'max\s*days\s*past\s*due'],
        'active_credit_cards_count': [r'عدد\s*بطاقات\s*الائتمان', r'active\s*credit\s*cards'],
    }
    for field, patterns in field_labels.items():
        for i, line in enumerate(lines):
            for pattern in patterns:
                m = re.search(pattern + r'\s*[:：-]?\s*(\d[\d,.،٬٫]*)\s*(?:جم|جنيه|EGP)?\s*$', line, re.I)
                if m:
                    value = _number(m.group(1))
                    if value is not None:
                        result[field] = int(value) if field in ('max_days_past_due', 'active_credit_cards_count') and value.is_integer() else value
                    break
                if re.fullmatch(pattern + r'\s*[:：-]?\s*', line, re.I) and i+1 < len(lines):
                    value = _number(lines[i+1])
                    if value is not None:
                        result[field] = int(value) if field in ('max_days_past_due', 'active_credit_cards_count') and value.is_integer() else value
                    break
            if result[field] is not None:
                break

    m = re.search(r'(?:نسبة\s*استخدام\s*البطاقات|card\s*utilization)\s*[:：-]?\s*(\d+(?:\.\d+)?)\s*(%)?', text, re.I)
    if m:
        value = float(m.group(1))
        ratio = value / 100 if m.group(2) or value > 1 else value
        if 0 <= ratio <= 1:
            result['total_credit_card_utilization'] = ratio
    return result


def _extract_via_ollama(text, model):
    from utils.ollama_client import ollama_extract
    prompt = (
        'Extract data from this Egyptian iScore OCR report. Return only JSON.\n'
        'Use exactly these top-level keys: ' + json.dumps(SCHEMA, ensure_ascii=False) + '\n'
        'Facility keys: ' + json.dumps(FACILITY_SCHEMA, ensure_ascii=False) + '\n'
        'Rules: Never guess. null for unsupported values, [] for uncertain facilities. '
        'Do not assign amounts by the order of OCR lines or confuse granted, used, overdue and installment. '
        'The report includes guarantor entries; do not treat guarantor entries as borrower debts. '
        'Do not use 300 or 850 as the score. ISO dates. Ratio utilization 0..1. '
        'If flattened columns make mapping ambiguous, return null.\nOCR:\n' + text[:6000]
    )
    return ollama_extract(prompt, model=model)


def _merge(result, candidate):
    if not isinstance(candidate, dict):
        return result
    money = {'total_active_loans_limit', 'total_outstanding_balance', 'total_overdue_amount'}
    ints = {'max_days_past_due', 'active_credit_cards_count'}
    for key in SCHEMA:
        if key == 'bureau_facilities' or result[key] is not None:
            continue
        value = candidate.get(key)
        if value is None:
            continue
        if key == 'score_date':
            value = _date(value)
        elif key == 'credit_score':
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 300 <= value <= 850:
                continue
            value = float(value)
        elif key in money | ints | {'total_credit_card_utilization'}:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                continue
            if key == 'total_credit_card_utilization' and value > 1:
                continue
            if key in ints:
                if not float(value).is_integer():
                    continue
                value = int(value)
        elif key == 'score_tier':
            continue
        if value is not None:
            result[key] = value
    # Deliberately do not automatically merge LLM facility rows from broken tables.
    return result


def extract(path: str, ollama_model: str = 'mistral:latest') -> dict:
    pages = load_document(path)
    text = ocr_pages(pages, lang='ara+eng')
    # debug_path = Path('iscore_ocr_debug.txt')
    # debug_path.write_text(text, encoding='utf-8')
    # print(f'    [iscore] OCR text saved to: {debug_path.resolve()}')
    print(f'    [iscore] OCR text length: {len(text)} characters')
    result = _extract_via_regex(text)
    missing = [k for k in ('credit_score', 'total_outstanding_balance') if result[k] is None]
    if missing:
        print(f'    [iscore] Regex missed {missing}; trying Ollama ({ollama_model})...')
        try:
            from utils.ollama_client import is_ollama_running
            if not is_ollama_running():
                raise RuntimeError('Ollama health check failed')
            result = _merge(result, _extract_via_ollama(text, ollama_model))
        except Exception as exc:
            print(f'    [iscore] Ollama fallback failed: {type(exc).__name__}: {exc}')
    # output = Path('iscore_result.json')
    # output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    # print(f'    [iscore] JSON saved to: {output.resolve()}')
    return result

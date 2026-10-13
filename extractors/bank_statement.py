"""
Bank Statement extractor.

Primary:  Tesseract OCR + transaction table parsing + computed metrics
Fallback: Ollama local LLM
"""

import re
import json
import statistics
from PIL import Image

from utils.pdf_to_images import load_document
from utils.ocr import ocr_pages


SCHEMA = {
    "bank_name": None,
    "account_number": None,
    "statement_period_months": None,
    "avg_monthly_net_inflow": None,
    "avg_monthly_balance": None,
    "min_monthly_balance": None,
    "max_monthly_balance": None,
    "balance_volatility_std": None,
    "overdraft_frequency": None,
    "returned_cheques_count": None,
    "income_regularity_score": None,
}

# Known Egyptian banks
KNOWN_BANKS = [
    "Mashreq", "CIB", "NBE", "البنك الأهلي", "QNB", "HSBC", "Banque Misr",
    "بنك مصر", "Arab Bank", "Faisal", "Alex Bank", "ADIB", "Emirates NBD",
    "Banque du Caire", "بنك القاهرة", "ABK", "Audi", "Societe Generale",
]


# ── money parsing ─────────────────────────────────────────────────────────────

def _to_float(s: str) -> float | None:
    s = s.replace(",", "").replace("،", "").replace(" ", "")
    # Handle Arabic-Indic digits
    ar_digits = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    s = s.translate(ar_digits)
    try:
        return float(s)
    except ValueError:
        return None


def _extract_all_amounts(text: str) -> list[float]:
    """Pull every number that looks like a balance/amount (>= 10)."""
    raw = re.findall(r"\b[\d,،٠-٩]{2,}(?:\.\d{1,2})?\b", text)
    amounts = []
    for r in raw:
        v = _to_float(r)
        if v is not None and v >= 10:
            amounts.append(v)
    return amounts


def _extract_balances(text: str) -> list[float]:
    """
    Try to extract running balance column values.
    Looks for lines that contain debit / credit / balance triplets.
    """
    balances = []
    # Pattern: a line with 2-3 numbers where the last looks like a running balance
    for line in text.splitlines():
        nums = _extract_all_amounts(line)
        if len(nums) >= 2:
            balances.append(nums[-1])   # last number on line = balance
    return balances


# ── regex extraction ──────────────────────────────────────────────────────────

def _extract_via_regex(text: str) -> dict:
    result = dict(SCHEMA)

    # Bank name
    for bank in KNOWN_BANKS:
        if bank.lower() in text.lower():
            result["bank_name"] = bank
            break

    # Account number
    for pattern in [
        r"(?:رقم الحساب|Account(?:\s+No\.?)?|Acc(?:ount)?\.?\s*No\.?)[:\s]*([\d\-]{8,20})",
        r"\b(01\d{9,14})\b",          # Mashreq-style
        r"\b(20\d{10,14})\b",         # Banque Misr style
        r"\b(\d{12,16})\b",           # generic long account number
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            result["account_number"] = m.group(1).strip()
            break

    # Overdraft / negative balance mentions
    overdraft_count = len(re.findall(
        r"(?:سحب على المكشوف|overdraft|مديون|DR\b)", text, re.IGNORECASE
    ))
    result["overdraft_frequency"] = overdraft_count

    # Returned cheques
    cheque_count = len(re.findall(
        r"(?:شيك مرتجع|returned cheque|عدم الصرف)", text, re.IGNORECASE
    ))
    result["returned_cheques_count"] = cheque_count

    # Statement period — look for explicit label first
    m = re.search(
        r"(?:statement period|from)[:\s]+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"
        r".{1,20}(?:to|till|until)[:\s]+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
        text, re.IGNORECASE
    )
    if m:
        from datetime import date as _date
        try:
            def _quick_date(s):
                for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y-%m-%d"):
                    try:
                        return datetime.strptime(s, fmt).date()
                    except ValueError:
                        continue
                return None
            d1 = _quick_date(m.group(1))
            d2 = _quick_date(m.group(2))
            if d1 and d2:
                months = abs((d2.year - d1.year) * 12 + d2.month - d1.month)
                result["statement_period_months"] = max(1, months)
        except Exception:
            pass

    # Fallback: count distinct year-months in the statement
    if result["statement_period_months"] is None:
        year_months = set(re.findall(r"\b(20\d{2})[/-](\d{1,2})", text))
        # Also match MM/YYYY style
        year_months |= set(
            (y, m) for m, y in re.findall(r"\b(\d{1,2})[/-](20\d{2})\b", text)
        )
        if len(year_months) >= 2:
            result["statement_period_months"] = len(year_months)

    # Balance metrics from table parsing
    balances = _extract_balances(text)
    if len(balances) >= 3:
        result["avg_monthly_balance"]    = round(statistics.mean(balances), 2)
        result["min_monthly_balance"]    = round(min(balances), 2)
        result["max_monthly_balance"]    = round(max(balances), 2)
        result["balance_volatility_std"] = round(statistics.stdev(balances), 2)

    # Income regularity score: look for credit amounts and compute CV
    # Credit lines typically say "إيداع" / "credit" / "CR"
    credit_amounts = []
    for line in text.splitlines():
        if re.search(r"(?:إيداع|credit|CR\b|مرتب|راتب)", line, re.IGNORECASE):
            nums = _extract_all_amounts(line)
            # Exclude account-number-sized values (> 100,000) and tiny noise (< 10)
            nums = [n for n in nums if 10 <= n <= 100000]
            if nums:
                credit_amounts.append(max(nums))

    if len(credit_amounts) >= 2:
        mean_c = statistics.mean(credit_amounts)
        std_c  = statistics.stdev(credit_amounts)
        cv     = std_c / mean_c if mean_c else 1.0
        result["income_regularity_score"] = round(float(max(0, 100 - cv * 100)), 2)
        result["avg_monthly_net_inflow"]  = round(float(mean_c), 2)
    elif len(credit_amounts) == 1:
        result["avg_monthly_net_inflow"]  = round(float(credit_amounts[0]), 2)
        result["income_regularity_score"] = 50.0  # single sample, unknown regularity

    return result


# ── Ollama fallback ───────────────────────────────────────────────────────────

def _extract_via_ollama(text: str, model: str) -> dict:
    from utils.ollama_client import ollama_extract

    prompt = f"""You are a financial analyst. Extract bank statement metrics from the OCR text below.
Return ONLY a valid JSON object matching this schema (null for missing values, numbers not strings):

{json.dumps(SCHEMA, ensure_ascii=False, indent=2)}

Rules:
- income_regularity_score: 0-100, higher = more consistent monthly credits
- avg_monthly_net_inflow: average of monthly credit totals
- balance_volatility_std: standard deviation of running balances
- overdraft_frequency: count of overdraft/negative events

OCR TEXT:
\"\"\"
{text[:5000]}
\"\"\"

Return ONLY the JSON object."""

    return ollama_extract(prompt, model=model)


# ── public entry point ────────────────────────────────────────────────────────

def extract(path: str, ollama_model: str = "qwen2.5:7b") -> dict:
    pages = load_document(path)
    text  = ocr_pages(pages, lang="ara+eng")

    print("    [bank_statement] Running Tesseract regex extraction…")
    result = _extract_via_regex(text)

    missing = [k for k in ("bank_name", "avg_monthly_balance", "income_regularity_score")
               if result.get(k) is None]

    if missing:
        print(f"    [bank_statement] Regex missed: {missing} — trying Ollama…")
        try:
            from utils.ollama_client import is_ollama_running
            if not is_ollama_running():
                raise RuntimeError("Ollama not running")
            ollama_result = _extract_via_ollama(text, model=ollama_model)
            numeric_fields = {
                "avg_monthly_net_inflow", "avg_monthly_balance",
                "min_monthly_balance", "max_monthly_balance",
                "balance_volatility_std", "income_regularity_score",
                "statement_period_months", "overdraft_frequency",
                "returned_cheques_count"
            }
            for k, v in ollama_result.items():
                if v is not None and result.get(k) is None:
                    if k in numeric_fields:
                        try:
                            result[k] = float(v) if k not in {
                                "statement_period_months",
                                "overdraft_frequency",
                                "returned_cheques_count"
                            } else int(float(v))
                        except (ValueError, TypeError):
                            result[k] = v
                    else:
                        result[k] = v
            print("    [bank_statement] ✓ Ollama filled gaps")
        except Exception as e:
            print(f"    [bank_statement] Ollama unavailable ({e}) — using regex result as-is")
    else:
        print("    [bank_statement] ✓ Regex extraction complete")

    return result
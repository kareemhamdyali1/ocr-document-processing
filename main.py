# main.py

import argparse
import json
import logging
import math
import sys
import uuid
from datetime import date, datetime
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger("document_pipeline")


# =========================================================
# General validation helpers
# =========================================================

def w(value, confidence=None):
    """
    Wrap a field in the existing value/confidence structure.

    Note:
        This is not calibrated OCR confidence. Until the extractors
        return a measured confidence, present values use a conservative
        placeholder rather than claiming 100% certainty.
    """
    if value is None:
        return {"value": None, "confidence": 0.0}

    if confidence is None:
        confidence = 0.5

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    return {
        "value": value,
        "confidence": confidence,
    }


def _clean_text(value, max_length=300):
    """Return normalized text or None."""
    if not isinstance(value, str):
        return None

    value = value.replace("\x00", "")
    value = " ".join(value.split())
    value = value.strip(" \t\r\n:;,-")

    if not value:
        return None

    if value.lower() in {
        "null",
        "none",
        "n/a",
        "na",
        "unknown",
        "not found",
        "not available",
    }:
        return None

    return value[:max_length]


def _normalize_digits(value):
    """Normalize Arabic-Indic and Persian digits to Western digits."""
    if not isinstance(value, str):
        return value

    translation = str.maketrans(
        "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
        "01234567890123456789",
    )
    return value.translate(translation)


def _number(
    value,
    minimum=None,
    maximum=None,
    integer=False,
):
    """
    Parse a finite numeric value.

    Returns None when the value is missing, malformed, infinite,
    or outside the permitted range.
    """
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, str):
        value = _normalize_digits(value.strip())
        value = (
            value.replace(",", "")
            .replace("،", "")
            .replace("٬", "")
            .replace("٫", ".")
            .strip()
        )

        if not value:
            return None

    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(parsed):
        return None

    if minimum is not None and parsed < minimum:
        return None

    if maximum is not None and parsed > maximum:
        return None

    if integer:
        if not parsed.is_integer():
            return None
        return int(parsed)

    return parsed


def _valid_iso_date(value, allow_future=False):
    """
    Validate a date formatted exactly as YYYY-MM-DD.

    Returns a datetime.date or None.
    """
    if not isinstance(value, str):
        return None

    value = value.strip()

    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None

    if parsed.isoformat() != value:
        return None

    if parsed.year < 1900:
        return None

    if not allow_future and parsed > date.today():
        return None

    return parsed


def _safe_date(value, allow_future=False):
    parsed = _valid_iso_date(value, allow_future=allow_future)
    return parsed.isoformat() if parsed else None


def _age(dob_string):
    dob = _valid_iso_date(dob_string)

    if dob is None:
        return None

    today = date.today()

    return (
        today.year
        - dob.year
        - ((today.month, today.day) < (dob.month, dob.day))
    )


def _safe_extractor_result(value, label):
    if not isinstance(value, dict):
        logger.error(
            "%s extractor returned %s instead of a dictionary.",
            label,
            type(value).__name__,
        )
        return {}

    return value


def _run_extractor(label, function, *args, **kwargs):
    """
    Run one extractor independently.

    If it fails, the rest of the pipeline continues and missing
    values remain null.
    """
    logger.info("Extracting %s...", label)

    try:
        result = function(*args, **kwargs)
        return _safe_extractor_result(result, label)

    except Exception:
        logger.exception(
            "%s extraction failed. Its fields will remain missing.",
            label,
        )
        return {}


def _safe_facilities(value):
    """Validate the shape and basic types of iScore facilities."""
    if not isinstance(value, list):
        return []

    facilities = []

    text_fields = (
        "lender_name",
        "facility_type",
        "status",
    )

    numeric_fields = {
        "granted_amount": (0, None, False),
        "outstanding_amount": (0, None, False),
        "installment_amount": (0, None, False),
        "overdue_days": (0, 10000, True),
        "overdue_amount": (0, None, False),
    }

    for item in value:
        if not isinstance(item, dict):
            continue

        facility = {}

        for key in text_fields:
            facility[key] = _clean_text(item.get(key), 160)

        for key, (minimum, maximum, integer) in numeric_fields.items():
            facility[key] = _number(
                item.get(key),
                minimum=minimum,
                maximum=maximum,
                integer=integer,
            )

        facility["start_date"] = _safe_date(
            item.get("start_date"),
            allow_future=False,
        )

        legal_action = item.get("legal_action_flag")

        if isinstance(legal_action, bool):
            facility["legal_action_flag"] = legal_action

        elif isinstance(legal_action, str):
            normalized = legal_action.strip().lower()

            if normalized in {"true", "yes", "1"}:
                facility["legal_action_flag"] = True
            elif normalized in {"false", "no", "0"}:
                facility["legal_action_flag"] = False
            else:
                facility["legal_action_flag"] = None

        else:
            facility["legal_action_flag"] = None

        # Ignore empty facility objects.
        if any(item is not None for item in facility.values()):
            facilities.append(facility)

    return facilities


def _completeness(fields, required_fields):
    """
    Return a completeness ratio, not an image quality score
    or a probability that the extraction is correct.
    """
    if not required_fields:
        return 0.0

    found = sum(fields.get(key) is not None for key in required_fields)

    return round(found / len(required_fields), 3)


def _document_summary(document_type, fields, required_fields):
    return {
        "document_type": document_type,
        "overall_quality_score": _completeness(
            fields,
            required_fields,
        ),
        # None means no actual tampering assessment was performed.
        "is_tampered_suspected": None,
        "extracted_fields": fields,
    }


# =========================================================
# Output assembler
# =========================================================

def assemble(nid, sal, bank, isc):
    nid = nid if isinstance(nid, dict) else {}
    sal = sal if isinstance(sal, dict) else {}
    bank = bank if isinstance(bank, dict) else {}
    isc = isc if isinstance(isc, dict) else {}

    application_id = (
        f"APP-{date.today().year}-EG-{uuid.uuid4().hex[:12].upper()}"
    )

    submission_timestamp = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    # -----------------------------------------------------
    # National ID
    # -----------------------------------------------------

    nid_name = _clean_text(
        nid.get("full_name") or nid.get("name_ar")
    )

    nid_number = _clean_text(
        nid.get("national_id") or nid.get("id_number"),
        max_length=14,
    )

    # Reject malformed IDs; do not silently modify them.
    if nid_number is not None:
        normalized_id = _normalize_digits(nid_number)
        if not (
            len(normalized_id) == 14
            and normalized_id.isdigit()
            and normalized_id[0] in {"2", "3"}
        ):
            logger.warning(
                "National ID has an unexpected format; returning it as missing."
            )
            nid_number = None
        else:
            nid_number = normalized_id

    nid_dob = _safe_date(
        nid.get("date_of_birth") or nid.get("birth_date")
    )

    nid_governorate = _clean_text(
        nid.get("governorate") or nid.get("place_of_birth")
    )

    nid_address = _clean_text(
        nid.get("address"),
        max_length=500,
    )

    # -----------------------------------------------------
    # Salary certificate
    # -----------------------------------------------------

    salary_fields = {
        "employer_name": _clean_text(sal.get("employer_name")),
        "employer_sector": _clean_text(sal.get("employer_sector")),
        "job_title": _clean_text(sal.get("job_title")),
        "declared_net_salary": _number(
            sal.get("declared_net_salary"),
            minimum=0,
        ),
        "declared_gross_salary": _number(
            sal.get("declared_gross_salary"),
            minimum=0,
        ),
        "employment_date": _safe_date(
            sal.get("employment_date")
        ),
        "employment_tenure_years": _number(
            sal.get("employment_tenure_years"),
            minimum=0,
            maximum=100,
        ),
        "issue_date": _safe_date(
            sal.get("issue_date")
        ),
    }

    if (
        salary_fields["declared_net_salary"] is not None
        and salary_fields["declared_gross_salary"] is not None
        and salary_fields["declared_net_salary"]
        > salary_fields["declared_gross_salary"] * 1.5
    ):
        logger.warning(
            "Net salary is much higher than gross salary. "
            "Check the original certificate."
        )

    # -----------------------------------------------------
    # Bank statement
    # -----------------------------------------------------

    bank_fields = {
        "bank_name": _clean_text(bank.get("bank_name")),
        "account_number": _clean_text(
            bank.get("account_number"),
            max_length=80,
        ),
        "statement_period_months": _number(
            bank.get("statement_period_months"),
            minimum=0,
            maximum=600,
        ),
        "avg_monthly_net_inflow": _number(
            bank.get("avg_monthly_net_inflow")
        ),
        "avg_monthly_balance": _number(
            bank.get("avg_monthly_balance")
        ),
        "min_monthly_balance": _number(
            bank.get("min_monthly_balance")
        ),
        "max_monthly_balance": _number(
            bank.get("max_monthly_balance")
        ),
        "balance_volatility_std": _number(
            bank.get("balance_volatility_std"),
            minimum=0,
        ),
        "overdraft_frequency": _number(
            bank.get("overdraft_frequency"),
            minimum=0,
            integer=True,
        ),
        "returned_cheques_count": _number(
            bank.get("returned_cheques_count"),
            minimum=0,
            integer=True,
        ),
        "income_regularity_score": _number(
            bank.get("income_regularity_score"),
            minimum=0,
            maximum=1,
        ),
    }

    # -----------------------------------------------------
    # iScore
    # -----------------------------------------------------

    iscore_fields = {
        "is_available": bool(isc),
        "credit_score": _number(
            isc.get("credit_score"),
            minimum=300,
            maximum=850,
            integer=True,
        ),
        "score_tier": _clean_text(
            isc.get("score_tier"),
            max_length=80,
        ),
        "score_date": _safe_date(
            isc.get("score_date")
        ),
        "total_active_loans_limit": _number(
            isc.get("total_active_loans_limit"),
            minimum=0,
        ),
        "total_outstanding_balance": _number(
            isc.get("total_outstanding_balance"),
            minimum=0,
        ),
        "total_overdue_amount": _number(
            isc.get("total_overdue_amount"),
            minimum=0,
        ),
        "max_days_past_due": _number(
            isc.get("max_days_past_due"),
            minimum=0,
            maximum=10000,
            integer=True,
        ),
        "active_credit_cards_count": _number(
            isc.get("active_credit_cards_count"),
            minimum=0,
            maximum=10000,
            integer=True,
        ),
        "total_credit_card_utilization": _number(
            isc.get("total_credit_card_utilization"),
            minimum=0,
            maximum=100,
        ),
        "bureau_facilities": _safe_facilities(
            isc.get("bureau_facilities")
        ),
    }

    # -----------------------------------------------------
    # Document summaries
    # -----------------------------------------------------

    documents = [
        _document_summary(
            "national_id",
            {
                "full_name": nid_name,
                "national_id": nid_number,
                "date_of_birth": nid_dob,
                "governorate": nid_governorate,
                "address": nid_address,
            },
            [
                "full_name",
                "national_id",
                "date_of_birth",
                "address",
            ],
        ),
        _document_summary(
            "salary_certificate",
            {
                "employer_name": salary_fields["employer_name"],
                "job_title": salary_fields["job_title"],
                "declared_net_salary": salary_fields["declared_net_salary"],
                "employment_date": salary_fields["employment_date"],
            },
            [
                "employer_name",
                "job_title",
                "declared_net_salary",
                "employment_date",
            ],
        ),
        _document_summary(
            "bank_statement",
            {
                "bank_name": bank_fields["bank_name"],
                "account_number": bank_fields["account_number"],
                "avg_monthly_net_inflow": bank_fields[
                    "avg_monthly_net_inflow"
                ],
                "avg_monthly_balance": bank_fields[
                    "avg_monthly_balance"
                ],
                "income_regularity_score": bank_fields[
                    "income_regularity_score"
                ],
            },
            [
                "bank_name",
                "account_number",
                "avg_monthly_net_inflow",
                "avg_monthly_balance",
            ],
        ),
        _document_summary(
            "iscore_report",
            {
                "credit_score": iscore_fields["credit_score"],
                "score_tier": iscore_fields["score_tier"],
                "score_date": iscore_fields["score_date"],
            },
            [
                "credit_score",
                "score_date",
            ],
        ),
    ]

    # -----------------------------------------------------
    # Final schema
    # -----------------------------------------------------

    result = {
        "application_id": application_id,
        "applicant_type": "individual",
        "submission_timestamp": submission_timestamp,
        "is_returning_customer": None,

        "documents": documents,

        "national_id_fields": {
            "full_name": w(nid_name),
            "national_id": w(nid_number),
            "date_of_birth": w(nid_dob),
            "age_years": w(_age(nid_dob)),
            "gender": w(_clean_text(nid.get("gender"), 40)),
            "governorate": w(nid_governorate),
            "address": w(nid_address),
            "profession_on_id": w(
                _clean_text(
                    nid.get("profession_on_id") or nid.get("job")
                )
            ),
            "marital_status_on_id": w(
                _clean_text(
                    nid.get("marital_status_on_id")
                    or nid.get("social_status")
                )
            ),
            "id_expiry_date": w(
                _safe_date(
                    nid.get("id_expiry_date"),
                    allow_future=True,
                )
            ),
        },

        "salary_certificate_fields": {
            key: w(value)
            for key, value in salary_fields.items()
        },

        "bank_statement_fields": {
            key: w(value)
            for key, value in bank_fields.items()
        },

        "iscore_report_fields": {
            "is_available": iscore_fields["is_available"],
            **{
                key: w(value)
                for key, value in iscore_fields.items()
                if key not in {"is_available", "bureau_facilities"}
            },
            "bureau_facilities": iscore_fields["bureau_facilities"],
        },

        "form_data": {
            "requested_amount": None,
            "tenure_months": None,
            "requested_annuity": None,
            "loan_purpose": None,
            "goods_price": 0.0,
            "family_status": None,
            "children_count": None,
            "family_members_count": None,
            "housing_type": None,
            "education_type": None,
            "owns_car": None,
            "owns_realty": None,
            "branch_id": None,
        },
    }

    return result


# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Egyptian document extraction pipeline"
    )

    parser.add_argument("--national_id_front", required=True)
    parser.add_argument("--national_id_back", required=True)
    parser.add_argument("--bank_statement", required=True)
    parser.add_argument("--salary_cert", required=True)
    parser.add_argument("--iscore", required=True)
    parser.add_argument("--output", default="result.json")
    parser.add_argument(
        "--ollama_model",
        default="qwen2.5:7b",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root))

    from extractors.national_id import extract as extract_nid
    from extractors.salary_cert import extract as extract_sal
    from extractors.bank_statement import extract as extract_bank
    from extractors.iscore import extract as extract_isc

    nid = _run_extractor(
        "National ID",
        extract_nid,
        args.national_id_front,
        args.national_id_back,
    )

    sal = _run_extractor(
        "Salary Certificate",
        extract_sal,
        args.salary_cert,
        ollama_model=args.ollama_model,
    )

    bank = _run_extractor(
        "Bank Statement",
        extract_bank,
        args.bank_statement,
        ollama_model=args.ollama_model,
    )

    isc = _run_extractor(
        "iScore Report",
        extract_isc,
        args.iscore,
        ollama_model=args.ollama_model,
    )

    logger.info("Assembling unified JSON...")
    result = assemble(nid, sal, bank, isc)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(
                result,
                file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            file.write("\n")

        temporary_path.replace(output_path)

    except Exception:
        logger.exception("Failed to save final JSON.")
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    logger.info("Completed. JSON saved to: %s", output_path)


if __name__ == "__main__":
    main()
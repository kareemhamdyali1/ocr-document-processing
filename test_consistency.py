from unittest import TestCase, main

from consistency import (
    build_consistency_checks,
    extract_identity_hints,
    _normalize_name,
)


NAME_A = (
    "\u0643\u0631\u064a\u0645 \u062d\u0645\u062f\u064a "
    "\u0639\u0644\u064a \u0627\u062d\u0645\u062f"
)

NAME_B = (
    "\u0643\u0631\u064a\u0645 \u062d\u0645\u062f\u0649 "
    "\u0639\u0644\u0649 \u0623\u062d\u0645\u062f"
)

NATIONAL_ID = "30306052103394"


class TestConsistency(TestCase):
    def sample(self):
        return {
            "national_id_fields": {
                "national_id": {"value": NATIONAL_ID},
                "name": {"value": NAME_A},
            },
            "salary_certificate_fields": {
                "employee_name": NAME_B,
                "national_id": NATIONAL_ID,
            },
            "bank_statement_fields": {
                "customer_name": NAME_A,
                "national_id": NATIONAL_ID,
            },
            "iscore_report_fields": {
                "national_id": NATIONAL_ID,
            },
        }

    def test_match(self):
        result = build_consistency_checks(self.sample())

        self.assertTrue(result["national_id_consistent"])
        self.assertTrue(result["name_consistent"])
        self.assertEqual(result["warnings"], [])

    def test_mismatch(self):
        payload = self.sample()

        payload["bank_statement_fields"]["national_id"] = (
            "29901011234567"
        )
        payload["salary_certificate_fields"]["employee_name"] = (
            "Different Person"
        )

        result = build_consistency_checks(payload)

        self.assertFalse(result["national_id_consistent"])
        self.assertFalse(result["name_consistent"])

    def test_missing_not_success(self):
        payload = {
            "national_id_fields": {},
            "salary_certificate_fields": {},
            "bank_statement_fields": {},
            "iscore_report_fields": {},
        }

        result = build_consistency_checks(payload)

        self.assertFalse(result["national_id_consistent"])
        self.assertFalse(result["name_consistent"])
        self.assertTrue(result["warnings"])

    def test_arabic_normalization(self):
        self.assertEqual(
            _normalize_name(NAME_A),
            _normalize_name(NAME_B),
        )

        payload = self.sample()
        payload["salary_certificate_fields"]["employee_name"] = NAME_A

        result = build_consistency_checks(payload)
        self.assertTrue(result["name_consistent"])

    def test_arabic_ocr_identity_hints(self):
        text = (
            "\u0627\u0633\u0645 \u0627\u0644\u0645\u0648\u0638\u0641: "
            + NAME_A
            + "\n"
            + "\u0627\u0644\u0631\u0642\u0645 \u0627\u0644\u0642\u0648\u0645\u064a: "
            + "\u0663\u0660\u0663\u0660\u0666\u0660\u0665\u0662"
            + "\u0661\u0660\u0663\u0663\u0669\u0664"
        )

        hints = extract_identity_hints(text, "salary_certificate")

        self.assertIn(NATIONAL_ID, hints["national_ids"])
        self.assertIn(NAME_A, hints["names"])


if __name__ == "__main__":
    main()

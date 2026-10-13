# OCR Document Processing

An AI-powered document processing pipeline designed to extract, normalize, and structure information from Egyptian financial and identity documents using OCR, text processing, and local LLM integration.

## Overview

This project provides a document extraction pipeline for processing multiple document types and converting unstructured document content into structured JSON outputs.

The system combines specialized document extractors, OCR utilities, validation logic, and optional local LLM assistance to improve information extraction and recovery.

## Features

- **Egyptian National ID Extraction:** Extract personal information, national ID number, date of birth, address, profession, gender, marital status, and card expiry date.
- **Salary Certificate Processing:** Extract employment and salary-related information from salary certificates.
- **Bank Statement Processing:** Process bank statement documents and extract relevant financial information.
- **iScore Report Processing:** Extract credit-report information for downstream financial analysis.
- **OCR Pipeline:** Integrate OCR engines and image-processing utilities for document text recognition.
- **LLM Integration:** Use local language models through Ollama for supported extraction and text-recovery tasks.
- **Structured JSON Output:** Convert extracted information into machine-readable JSON.
- **Confidence Assessment:** Evaluate extraction confidence using dedicated confidence-processing utilities.
- **FastAPI Integration:** Provide an API layer for integrating document processing into other applications.

## Supported Documents

| Document | Extraction Focus |
|---|---|
| Egyptian National ID | Identity and personal information |
| Salary Certificate | Salary and employment information |
| Bank Statement | Financial transactions and account information |
| iScore Report | Credit facilities and credit-related information |

*Extracted fields and availability depend on document quality, layout, and the corresponding extractor implementation.*

## Tech Stack

- Python
- FastAPI
- OCR and image-processing tools
- Ollama for local LLM inference
- JSON for structured extraction results
- Git and GitHub

## Project Structure

```text
ocr-document-processing/
├── api.py
├── main.py
├── confidence.py
├── ollama_client.py
├── requirements.txt
├── extractors/
│   ├── __init__.py
│   ├── national_id.py
│   ├── salary_cert.py
│   ├── bank_statement.py
│   └── iscore.py
├── utils/
│   ├── __init__.py
│   ├── ocr.py
│   ├── ollama_client.py
│   └── pdf_to_images.py
├── vendor/
│   └── Egyptian-National-ID-Card-OCR-System/
└── .gitignore
```

## Requirements

Before running the project, install:

- Python compatible with the dependencies in `requirements.txt`
- Tesseract OCR, if required by the selected extraction pipeline
- Ollama and the configured model, if using local LLM functionality

Some OCR components may require additional language data or system-level dependencies.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/kareemhamdyali1/ocr-document-processing.git
cd ocr-document-processing
```

### 2. Create a virtual environment

**Windows — PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure OCR and local models

Make sure required OCR engines and language data are installed and accessible.

If the pipeline uses Ollama, install and start Ollama, download the model configured by your application, and verify that the model is available locally.

For example:

```bash
ollama list
```

Model requirements depend on the configuration and the available system resources.

## Running the Application

### Run the main pipeline

```bash
python main.py
```

The exact processing arguments depend on the CLI implementation in `main.py`.

### Start the FastAPI server

If `api.py` exposes the FastAPI application as `app`, start it using:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Once the server starts, open:

- **Swagger UI:** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc

Use the API documentation to confirm the available endpoints, required upload fields, and response schema.

## Example JSON Output

The following example illustrates the general structure of an extracted Egyptian National ID. Values are illustrative and are not intended to represent a real document.

```json
{
  "full_name": "اسم افتراضي",
  "national_id": "00000000000000",
  "date_of_birth": "2000-01-01",
  "governorate": "القاهرة",
  "address": "عنوان افتراضي",
  "profession_on_id": "المهنة",
  "gender": "ذكر",
  "marital_status_on_id": "أعزب",
  "id_expiry_date": "2030-01-01",
  "age_years": 26
}
```

The actual response fields and values depend on the document extractor and the input document.

## Configuration and Customization

The extraction pipeline can be extended to support additional document types, OCR engines, validation rules, and model-assisted extraction.

Recommended configuration practices:

- Store API keys and secrets in environment variables.
- Avoid hardcoding machine-specific file paths.
- Keep document samples and generated results out of version control.
- Configure OCR language data and model dependencies for the target environment.
- Validate extracted fields before using them in downstream financial decisions.

## Limitations

- OCR accuracy depends on image quality, resolution, orientation, and document layout.
- Arabic text recognition may require specialized preprocessing and language data.
- LLM-assisted extraction can produce incorrect or incomplete values and should be validated.
- Expired dates or missing fields should not be treated as valid without verification.
- Extracted data should be reviewed before being used for credit or other consequential decisions.

## Security and Privacy

This project may process sensitive identity and financial information.

- Do not commit real identity documents, bank statements, or credit reports.
- Never publish API keys, credentials, or private configuration files.
- Restrict access to uploaded documents and generated JSON files.
- Review third-party code and OCR language-data licenses before redistributing them.
- Keep the repository private until the redistribution permissions for bundled components have been verified.

## Future Improvements

Potential areas for further development include:

- Improved Arabic OCR accuracy and field-level validation
- More robust document classification and extraction
- Consistent confidence scoring across document types
- Automated testing with anonymized sample documents
- Production deployment and monitoring
- Better error handling and structured API responses

## License

The licensing terms for this project and its bundled third-party components must be verified before redistribution. Third-party components may be subject to separate licenses.

## Author

**Kareem Hamdy Ali**

AI Engineer | Document AI | OCR | LLM Integration

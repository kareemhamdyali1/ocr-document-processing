"""Local FastAPI adapter for the existing OCR pipeline.

Uploaded files exist only in a temporary directory during the request.
The API does not persist the assembled JSON. Individual extractors may
still create their own debug/output files and must be configured separately.
"""
import logging
import tempfile
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from main import _run_extractor, assemble
from confidence import apply_validation_scores
from extractors.national_id import extract as extract_nid
from extractors.salary_cert import extract as extract_sal
from extractors.bank_statement import extract as extract_bank
from extractors.iscore import extract as extract_isc

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_SUFFIXES = {'.jpg', '.jpeg', '.png', '.pdf', '.webp', '.tif', '.tiff'}
PROCESS_LOCK = Lock()
logger = logging.getLogger('ocr_api')
app = FastAPI(title='Egyptian Document OCR API', version='1.1.0')


@app.get('/health')
def health():
    return {'status': 'ok', 'service': 'egyptian-document-ocr'}


def save_upload(upload: UploadFile, directory: Path, field_name: str) -> Path:
    suffix = Path(upload.filename or '').suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f'Unsupported file type for {field_name}: {suffix or "none"}')
    destination = directory / f'{field_name}{suffix}'
    size = 0
    try:
        with destination.open('wb') as out:
            while True:
                chunk = upload.file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail=f'{field_name} exceeds 20 MB')
                out.write(chunk)
    finally:
        upload.file.close()
    if not size:
        raise HTTPException(status_code=400, detail=f'{field_name} is empty')
    return destination


def process(paths: dict[str, Path], ollama_model: str):
    with PROCESS_LOCK:
        nid = _run_extractor('National ID', extract_nid, str(paths['national_id_front']), str(paths['national_id_back']))
        sal = _run_extractor('Salary Certificate', extract_sal, str(paths['salary_cert']), ollama_model=ollama_model)
        bank = _run_extractor('Bank Statement', extract_bank, str(paths['bank_statement']), ollama_model=ollama_model)
        isc = _run_extractor('iScore Report', extract_isc, str(paths['iscore']), ollama_model=ollama_model)
        return apply_validation_scores(assemble(nid, sal, bank, isc))


@app.post('/api/v1/ocr/process-documents')
async def process_documents(
    national_id_front: UploadFile = File(...),
    national_id_back: UploadFile = File(...),
    salary_cert: UploadFile = File(...),
    bank_statement: UploadFile = File(...),
    iscore: UploadFile = File(...),
    ollama_model: str = Form('mistral'),
):
    uploads = {
        'national_id_front': national_id_front,
        'national_id_back': national_id_back,
        'salary_cert': salary_cert,
        'bank_statement': bank_statement,
        'iscore': iscore,
    }
    if not ollama_model.strip():
        raise HTTPException(status_code=400, detail='ollama_model must not be empty')
    with tempfile.TemporaryDirectory(prefix='ocr_upload_') as temp:
        paths = {}
        for name, upload in uploads.items():
            paths[name] = await run_in_threadpool(save_upload, upload, Path(temp), name)
        try:
            return await run_in_threadpool(process, paths, ollama_model.strip())
        except Exception:
            logger.exception('OCR request failed')
            raise HTTPException(status_code=500, detail='OCR processing failed; check server logs')

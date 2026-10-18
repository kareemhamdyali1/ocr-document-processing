
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-ara \
    tesseract-ocr-eng \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .


RUN python -c "import skimage; print('scikit-image OK')" \
    && which tesseract \
    && tesseract --version \
    && test -f /usr/share/tesseract-ocr/5/tessdata/ara.traineddata \
    && test -f /usr/share/tesseract-ocr/5/tessdata/eng.traineddata \
    && TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata tesseract --list-langs
    
COPY vendor/Egyptian-National-ID-Card-OCR-System/ara_number.traineddata \
     /usr/share/tesseract-ocr/5/tessdata/ara_number.traineddata
COPY vendor/Egyptian-National-ID-Card-OCR-System/ara_combined.traineddata \
     /usr/share/tesseract-ocr/5/tessdata/ara_combined.traineddata

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080}"]

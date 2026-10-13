"""
Egyptian National ID Card Recognition Web Service

This FastAPI application provides a web service for processing Egyptian national ID cards.
It offers both a web interface and REST API endpoints for uploading card images,
extracting text data, and returning structured JSON results.

Features:
- Upload front and back images of Egyptian ID cards
- Automatic card detection and perspective correction
- OCR text extraction in Arabic
- Structured JSON output with all card fields
- Web interface for easy interaction
- Session-based result storage

API Endpoints:
- POST /api/upload: Upload card images and get extracted data
- GET /api/result/{session_id}: Retrieve processing results
- GET /api/temp/{image_type}: Get processed card images
- GET /: Serve web interface

Usage:
    # As web server
    uvicorn main:app --reload --port 8000

    # As command line tool
    python main.py --front front_image.jpg --back back_image.jpg
"""

import os
import json
import uuid
import datetime
from typing import Dict, Any
import sys
import argparse

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import card recognition module for image processing
try:
    import card_recognition
except ImportError:
    card_recognition = None
    print("[Warning] card_recognition module not available - image processing disabled")

# Import OCR data extraction module
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("ocr_extractor", "id_card_data_extractor.py.py")
    ocr_extractor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ocr_extractor)
except Exception as e:
    ocr_extractor = None
    print(f"[Warning] OCR data extraction module not available: {e}")

# Directory configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Ensure required directories exist
for directory in (STATIC_DIR, UPLOAD_DIR, RESULTS_DIR):
    try:
        os.makedirs(directory, exist_ok=True)
    except FileExistsError:
        pass

INDEX_HTML = os.path.join(STATIC_DIR, "index.html")

# FastAPI application setup
app = FastAPI(
    title="Egyptian National ID Recognition Service",
    description="Web service for processing Egyptian national ID cards",
    version="1.0.0"
)

# Configure CORS for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (web interface)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Default field structure for Egyptian ID cards
DEFAULT_FIELDS = {
    "id_number": None,        # 14-digit national ID number
    "name_ar": None,          # Full name in Arabic
    "address": None,          # Address in Arabic
    "birth_date": None,       # Date of birth (YYYY/MM/DD)
    "gender": None,           # Gender (ذكر/أنثى)
    "religion": None,         # Religion (مسلم/مسيحي)
    "social_status": None,    # Marital status (أعزب/متزوج/etc.)
    "husband_name": None,     # Husband's name (for married women)
    "job": None,              # Occupation
    "place_of_work": None,    # Workplace
    "place_of_birth": None,   # Governorate of birth
}

def _safe_extract(front_path: str, back_path: str) -> Dict[str, Any]:
    """
    Extract data from card images using card_recognition and OCR extraction modules.

    Args:
    - front_path (str): Path to front card image
    - back_path (str): Path to back card image

    Returns:
    - Dict[str, Any]: Extracted data in DEFAULT_FIELDS format
    """
    data: Dict[str, Any] = {}

    if card_recognition is not None and ocr_extractor is not None:
        try:
            # First scan the images to get processed versions
            card_recognition.scan_id_card(front_path, "front")
            card_recognition.scan_id_card(back_path, "back")

            # Then use OCR extractor to extract data from the processed images
            ocr_extractor.extract_front_side_data("temp_front.jpg")
            ocr_extractor.extract_back_side_data("temp_back.jpg")

            # Extract data from OCR extractor's global variables
            husband_name = getattr(ocr_extractor, 'extracted_spouse_name', '')
            place_of_work = getattr(ocr_extractor, 'extracted_workplace', '')

            data = {
                "id_number": getattr(ocr_extractor, 'extracted_id_number', ''),
                "name_ar": getattr(ocr_extractor, 'extracted_full_name', ''),
                "address": getattr(ocr_extractor, 'extracted_address', ''),
                "birth_date": getattr(ocr_extractor, 'extracted_birth_date', ''),
                "gender": getattr(ocr_extractor, 'extracted_gender', ''),
                "religion": getattr(ocr_extractor, 'extracted_religion', ''),
                "social_status": getattr(ocr_extractor, 'extracted_marital_status', ''),
                "husband_name": husband_name if husband_name and len(husband_name.strip()) > 1 else "لا يوجد",
                "job": getattr(ocr_extractor, 'extracted_job_title', ''),
                "place_of_work": place_of_work if place_of_work and len(place_of_work.strip()) > 1 else "لا يوجد",
                "place_of_birth": getattr(ocr_extractor, 'extracted_governorate', ''),
            }

        except Exception as e:
            print(f"Error during extraction: {e}")
            pass

    # Ensure keys match DEFAULT_FIELDS
    return {k: data.get(k) for k in DEFAULT_FIELDS.keys()}

@app.get("/health")
def health():
    """
    Health check endpoint.

    Returns:
    - Dict[str, Any]: Health status
    """
    return {"ok": True, "ts": datetime.datetime.utcnow().isoformat() + "Z"}

@app.get("/")
def root():
    """
    Serve web interface.

    Returns:
    - HTMLResponse: Web interface
    """
    if os.path.exists(INDEX_HTML):
        return FileResponse(INDEX_HTML, media_type="text/html")
    # Fallback message if index.html not found
    return HTMLResponse("<h1>Egyptian National ID API</h1><p>Place your frontend at ./static/index.html</p>")

@app.post("/api/upload")
async def upload(front: UploadFile = File(...), back: UploadFile = File(...)):
    """
    Upload card images and get extracted data.

    Args:
    - front (UploadFile): Front card image
    - back (UploadFile): Back card image

    Returns:
    - JSONResponse: Extracted data
    """
    if front.content_type and not front.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Front file must be an image")
    if back.content_type and not back.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Back file must be an image")

    sid = uuid.uuid4().hex[:12]
    session_dir = os.path.join(RESULTS_DIR, sid)
    os.makedirs(session_dir, exist_ok=True)

    front_path = os.path.join(session_dir, f"front_{sid}.jpg")
    back_path  = os.path.join(session_dir, f"back_{sid}.jpg")

    # Save uploads
    with open(front_path, "wb") as f: f.write(await front.read())
    with open(back_path,  "wb") as f: f.write(await back.read())

    # Extract fields (optional if card_recognition provided)
    fields = _safe_extract(front_path, back_path)

    payload = {
        "session_id": sid,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "images": {"front": os.path.basename(front_path), "back": os.path.basename(back_path)},
        "fields": {**DEFAULT_FIELDS, **fields},
        "source": "card_recognition-fastapi",
        "version": "1.0",
    }

    out_path = os.path.join(session_dir, f"result_{sid}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return JSONResponse(payload)

@app.get("/api/result/{session_id}")
def get_result(session_id: str):
    """
    Retrieve processing results.

    Args:
    - session_id (str): Session ID

    Returns:
    - JSONResponse: Processing results
    """
    session_dir = os.path.join(RESULTS_DIR, session_id)
    out_path = os.path.join(session_dir, f"result_{session_id}.json")
    if not os.path.exists(out_path):
        raise HTTPException(status_code=404, detail="Result not found")
    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(data)

@app.get("/api/result/{session_id}/download")
def download_json(session_id: str):
    """
    Download processing results as JSON.

    Args:
    - session_id (str): Session ID

    Returns:
    - FileResponse: JSON file
    """
    session_dir = os.path.join(RESULTS_DIR, session_id)
    out_path = os.path.join(session_dir, f"result_{session_id}.json")
    if not os.path.exists(out_path):
        raise HTTPException(status_code=404, detail="Result not found")
    return FileResponse(out_path, media_type="application/json", filename=f"national_id_{session_id}.json")

@app.get("/api/temp/{image_type}")
def get_temp_image(image_type: str):
    """
    Serve processed temp images (temp_front.jpg or temp_back.jpg).

    Args:
    - image_type (str): Image type (front/back)

    Returns:
    - FileResponse: Processed image
    """
    if image_type not in ["front", "back"]:
        raise HTTPException(status_code=400, detail="Invalid image type")

    temp_file = f"temp_{image_type}.jpg"
    if not os.path.exists(temp_file):
        raise HTTPException(status_code=404, detail="Processed image not found")

    return FileResponse(temp_file, media_type="image/jpeg")


# ==================== STANDALONE MODE ====================

def extract_data_from_images(front_path: str, back_path: str) -> Dict[str, Any]:
    """
    Extract data from card images for standalone mode.

    Args:
        front_path (str): Path to front card image
        back_path (str): Path to back card image

    Returns:
        Dict[str, Any]: Extracted card data
    """
    print(f"Processing front image: {front_path}")
    print(f"Processing back image: {back_path}")

    # Extract fields using the same logic as the web API
    fields = _safe_extract(front_path, back_path)

    # Create JSON output similar to web API
    result = {
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "images": {
            "front": os.path.basename(front_path),
            "back": os.path.basename(back_path)
        },
        "fields": {**DEFAULT_FIELDS, **fields},
        "source": "card_recognition-cli",
        "version": "1.0",
    }

    return result


def save_results_to_file(data: Dict[str, Any]) -> str:
    """
    Save extraction results to a timestamped JSON file.

    Args:
        data (Dict[str, Any]): Extraction results

    Returns:
        str: Output filename
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"result_{timestamp}.json"

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return output_filename


def run_standalone_mode(front_image: str, back_image: str) -> None:
    """
    Run the application in standalone command-line mode.

    Args:
        front_image (str): Path to front card image
        back_image (str): Path to back card image
    """
    try:
        # Extract data from images
        results = extract_data_from_images(front_image, back_image)

        # Display results to console
        print("\n" + "="*50)
        print("EXTRACTION RESULTS")
        print("="*50)
        print(json.dumps(results, ensure_ascii=False, indent=2))

        # Save results to file
        output_file = save_results_to_file(results)
        print(f"\n[SUCCESS] Results saved to: {output_file}")
        print("Processing completed successfully!")

    except Exception as e:
        print(f"[ERROR] Failed to process images: {e}")
        sys.exit(1)


def run_web_server() -> None:
    """
    Run the application as a web server.
    """
    print("Starting Egyptian National ID Recognition Web Service...")
    print("Web interface: http://localhost:8000")
    print("API documentation: http://localhost:8000/docs")
    print("Use --front and --back arguments to process images directly")

    try:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    except ImportError:
        print("[ERROR] uvicorn not installed. Install with: pip install uvicorn")
        sys.exit(1)


# ==================== MAIN ENTRY POINT ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Egyptian National ID Card Recognition Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run as web server
  python main.py

  # Process images directly
  python main.py --front front_card.jpg --back back_card.jpg
        """
    )

    parser.add_argument(
        "--front",
        help="Path to front card image file",
        metavar="IMAGE_PATH"
    )
    parser.add_argument(
        "--back",
        help="Path to back card image file",
        metavar="IMAGE_PATH"
    )

    args = parser.parse_args()

    # Check if both front and back images are provided for standalone mode
    if args.front and args.back:
        # Validate image files exist
        if not os.path.exists(args.front):
            print(f"[ERROR] Front image file not found: {args.front}")
            sys.exit(1)
        if not os.path.exists(args.back):
            print(f"[ERROR] Back image file not found: {args.back}")
            sys.exit(1)

        # Run in standalone mode
        run_standalone_mode(args.front, args.back)

    elif args.front or args.back:
        # Only one image provided - show error
        print("[ERROR] Both --front and --back images are required for processing")
        parser.print_help()
        sys.exit(1)

    else:
        # No arguments provided - run as web server
        run_web_server()

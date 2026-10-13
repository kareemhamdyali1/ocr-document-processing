# Egyptian National ID Card OCR System

A comprehensive web service for processing and extracting data from Egyptian national ID cards using advanced OCR (Optical Character Recognition) and computer vision techniques.

**Developed by:**
- [Marwan Ghazal](https://github.com/marwan-ghazal)
- [Omar Amin](https://github.com/omaramin-77)

##  Features

- **Automatic Card Detection**: Detects and extracts ID cards from photos using edge detection and contour analysis
- **Perspective Correction**: Automatically corrects perspective distortion for optimal OCR accuracy
- **Dual-Side Processing**: Processes both front and back sides of Egyptian national ID cards
- **Arabic OCR Support**: Specialized Arabic text recognition with Tesseract OCR
- **Data Validation**: Validates extracted ID numbers using Egyptian ID format rules
- **Web Interface**: Modern, responsive web interface for easy card scanning
- **REST API**: RESTful API endpoints for programmatic access
- **Real-time Processing**: Fast processing with visual feedback

##  Extracted Information

### Front Side
- **Personal Photo**: Extracted ID photo
- **Full Name**: Arabic name recognition
- **Address**: Complete address in Arabic
- **National ID Number**: 14-digit Egyptian ID number
- **Birth Date**: Calculated from ID number
- **Governorate**: Place of birth from ID number

### Back Side
- **Job Title**: Occupation/profession
- **Workplace**: Employer information
- **Gender**: Male/Female (derived from ID number)
- **Religion**: Religious affiliation
- **Marital Status**: Single/Married status
- **Spouse Name**: Partner's name (if applicable)

##  Project Structure

```
ocr-arabic-master1/
├── main.py                           # FastAPI web server and API endpoints
├── card_recognition.py               # Image processing and card detection
├── id_card_data_extractor.py.py     # OCR text extraction and validation
├── transform.py                      # Perspective correction algorithms
├── gender.py                         # Gender detection from ID number
├── pob.py                           # Place of birth mapping
├── static/
│   └── index.html                   # Web interface
├── uploads/                         # Temporary upload storage
├── results/                         # Processing results
└── README.md                        # This file
```

##  Core Modules

### `main.py`
- **FastAPI Application**: Web server setup and configuration
- **API Endpoints**: RESTful endpoints for card processing
- **File Management**: Upload handling and result storage
- **CORS Configuration**: Cross-origin resource sharing setup

### `card_recognition.py`
- **Image Processing**: Load, resize, and preprocess images
- **Edge Detection**: Canny edge detection with adaptive thresholds
- **Contour Analysis**: Find and validate quadrilateral card shapes
- **OCR Validation**: Validate card candidates using ID number OCR
- **Perspective Correction**: Apply geometric transformations

### `id_card_data_extractor.py.py`
- **Front Side OCR**: Extract name, address, and ID number
- **Back Side OCR**: Extract job, religion, and marital status
- **Text Normalization**: Handle Arabic text variations
- **Data Validation**: Validate and clean extracted text
- **Birth Date Calculation**: Derive birth date from ID format

### `transform.py`
- **Corner Sorting**: Order quadrilateral corners consistently
- **Perspective Transformation**: Convert distorted images to rectangular view
- **Geometric Calculations**: Compute optimal output dimensions
- **Image Warping**: Apply perspective correction matrix

### `gender.py`
- **Gender Detection**: Determine gender from ID number digit
- **ID Validation**: Validate ID number format and length

### `pob.py`
- **Governorate Mapping**: Map ID codes to Egyptian governorates
- **Place of Birth**: Determine birthplace from ID number

##  Installation

### Prerequisites

1. **Python 3.8+**
2. **Tesseract OCR** with Arabic language support
3. **Required Python packages** (see requirements below)

### System Dependencies

#### Windows
```bash
# Install Tesseract OCR
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH: C:\Program Files\Tesseract-OCR

# Install Arabic language data
# Download ara.traineddata to tessdata folder
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-ara
sudo apt install python3-opencv
```

#### macOS
```bash
brew install tesseract tesseract-lang
```

### Python Dependencies

Install with:
```bash
pip install -r requirements.txt
```

##  Usage

### Starting the Web Server

```bash
# Navigate to project directory
cd Egyptian-National-ID-Card-OCR-System
# Start the FastAPI server
uvicorn main:app --reload --port 8000

# Or use Python directly
python -m uvicorn main:app --reload --port 8000
```

### Web Interface

1. Open browser to `http://localhost:8000`
2. Upload front and back images of Egyptian ID card
3. Click "Process Images" to extract data
4. View extracted information in structured format
5. Download results as JSON file

### API Endpoints

#### Upload and Process ID Card
```http
POST /api/upload
Content-Type: multipart/form-data

Parameters:
- front: Image file (front side of ID)
- back: Image file (back side of ID)

Response:
{
  "session_id": "unique_session_id",
  "created_at": "2024-01-01T12:00:00Z",
  "images": {
    "front": "front_session_id.jpg",
    "back": "back_session_id.jpg"
  },
  "fields": {
    "id_number": "29607220103693",
    "name_ar": "أحمد محمد علي",
    "address": "شارع النيل، القاهرة",
    "birth_date": "1996/07/22",
    "gender": "ذكر",
    "religion": "مسلم",
    "social_status": "أعزب",
    "husband_name": "لا يوجد",
    "job": "مهندس",
    "place_of_work": "شركة التكنولوجيا",
    "place_of_birth": "القاهرة"
  },
  "source": "card_recognition-fastapi",
  "version": "1.0"
}
```

#### Get Processed Images
```http
GET /api/temp/front?t=timestamp
GET /api/temp/back?t=timestamp
```

#### Health Check
```http
GET /health
```

##  Configuration

### Tesseract Configuration

Update paths in `id_card_data_extractor.py.py`:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"
```

### Processing Parameters

Adjust in `card_recognition.py`:
```python
PROCESSING_HEIGHT = 500          # Image processing height
FINAL_CARD_SIZE = (1000, 630)    # Output card dimensions
EXPECTED_ASPECT_RATIO = 1.58      # ID card aspect ratio
```

##  Testing

### Manual Testing
```bash
# Test with sample images
curl -X POST "http://localhost:8000/api/upload" \
  -F "front=@sample_front.jpg" \
  -F "back=@sample_back.jpg"
```

### Command Line Usage
```python
# Direct module usage
from card_recognition import scan_id_card
from id_card_data_extractor import extract_front_side_data, extract_back_side_data

# Process images
scan_id_card("front.jpg", "front")
scan_id_card("back.jpg", "back")

# Extract data
extract_front_side_data("temp_front.jpg")
extract_back_side_data("temp_back.jpg")
```

##  Troubleshooting

### Common Issues

1. **Tesseract Not Found**
   ```
   Error: Tesseract not installed or not in PATH
   Solution: Install Tesseract and add to system PATH
   ```

2. **Arabic OCR Poor Quality**
   ```
   Issue: Extracted Arabic text is incorrect
   Solution: Ensure ara.traineddata is installed in tessdata folder
   ```

3. **Card Not Detected**
   ```
   Issue: No quadrilateral contour found
   Solution: Ensure good lighting and clear card boundaries
   ```

4. **Import Errors**
   ```
   Issue: Module not found errors
   Solution: Check file extensions and import paths
   ```

### Debug Mode

Enable debug output in `card_recognition.py`:
```python
# Uncomment debug lines to see processing steps
cv2.imshow("Debug", debug_image)
cv2.waitKey(0)
```

**Note**: This system is designed specifically for Egyptian national ID cards and may require modifications for other document types or countries.

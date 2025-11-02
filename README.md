# Extract Paddle - PaddleOCR Document Extraction Service

A Flask microservice for extracting structured data from PDFs and images using PaddleOCR layout parsing API.

## Features

- **PDF and Image Support**: Process PDFs and common image formats (PNG, JPG, JPEG, BMP, TIFF)
- **Structured Output**: Returns `storage_data` format with source blocks, markdown, and metadata
- **Extraction Timing**: Includes extraction time in response metadata
- **REST API**: Simple Flask-based API for easy integration
- **Error Handling**: Comprehensive error handling and logging

## Installation

1. Create and activate virtual environment:

```bash
python3.11 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your configuration
```

## Configuration

Environment variables (see `.env.example`):

- `PADDLEOCR_API_URL`: PaddleOCR API endpoint
- `PADDLEOCR_API_TOKEN`: API authentication token
- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `5002`)
- `DEBUG`: Enable debug mode (default: `false`)
- `MAX_FILE_SIZE`: Maximum file size in bytes (default: `52428800` = 50MB)

## Usage

### Start the server:

```bash
python app.py
```

Or with gunicorn (production):

```bash
gunicorn -w 1 -b 0.0.0.0:5002 app:app
```

The service will run on `http://localhost:5002`

### API Endpoints

#### 1. Health Check

```bash
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "service": "extract-paddle",
  "api_url": "..."
}
```

#### 2. Extract Document

```bash
POST /extract
Content-Type: multipart/form-data
Body: file (PDF or image file)
```

**Response:**

```json
{
  "documentId": "document_name",
  "extractionMetadata": {
    "logId": "...",
    "timestamp": "2024-01-01T00:00:00",
    "fileType": "pdf",
    "numPages": 13,
    "extractionTimeSeconds": 75.42
  },
  "pages": [
    {
      "pageIndex": 0,
      "pageWidth": 792,
      "pageHeight": 612,
      "sourceBlocks": [...],
      "markdown": {
        "text": "...",
        "images": {...}
      },
      "layoutBoxes": [...]
    }
  ]
}
```

### Example: Using curl

```bash
curl -X POST http://localhost:5002/extract \
  -F "file=@document.pdf" \
  -H "Content-Type: multipart/form-data"
```

### Example: Using Python requests

```python
import requests

with open('document.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:5002/extract',
        files={'file': f}
    )

result = response.json()
print(f"Extracted {result['extractionMetadata']['numPages']} pages")
print(f"Extraction took {result['extractionMetadata']['extractionTimeSeconds']} seconds")
```

## Response Format

The service returns `storage_data` structure:

- **documentId**: Identifier derived from filename
- **extractionMetadata**:
  - `logId`: API request ID
  - `timestamp`: ISO timestamp
  - `fileType`: "pdf" or image type
  - `numPages`: Number of pages
  - `extractionTimeSeconds`: Total extraction time
- **pages**: Array of page data with:
  - `pageIndex`: Page number (0-based)
  - `pageWidth`/`pageHeight`: Page dimensions
  - `sourceBlocks`: Structured blocks with content, labels, bounding boxes
  - `markdown`: Markdown representation of page content
  - `layoutBoxes`: Layout detection boxes

## Error Handling

The service returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad request (missing file, invalid file type, file too large)
- `500`: Internal server error
- `504`: Gateway timeout (API request timeout)

Error responses include:

```json
{
  "error": "Error type",
  "message": "Detailed error message"
}
```

## Development

### Running in Development Mode:

```bash
export DEBUG=true
python app.py
```

### Logging

The service uses Python's logging module with INFO level by default. Logs include:

- File processing status
- API request timing
- Error details with stack traces

## Notes

- The service uses the PaddleOCR layout parsing API (not local inference)
- Files are temporarily stored during processing and automatically cleaned up
- Maximum file size is configurable via `MAX_FILE_SIZE` environment variable
- Extraction time includes file I/O, API call, and data processing

## License

See parent project license.
# text-extract-with-paddle-ocr

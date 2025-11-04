"""
Flask server for PaddleOCR document extraction.

Accepts PDF or image files and returns structured extraction data.
"""

import os
import base64
import time
import logging
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from utils.extractors import extract_storage_data
import requests

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Configuration from environment variables
API_URL = os.getenv(
    "PADDLEOCR_API_URL",
    "https://h0p83afbb9e7r7t5.aistudio-app.com/layout-parsing",
)
API_TOKEN = os.getenv("PADDLEOCR_API_TOKEN")
if not API_TOKEN:
    raise ValueError(
        "PADDLEOCR_API_TOKEN not found in environment variables. "
        "Please set it in your .env file."
    )
max_file_size_str = os.getenv("MAX_FILE_SIZE")
if not max_file_size_str:
    raise ValueError(
        "MAX_FILE_SIZE not found in environment variables. "
        "Please set it in your .env file."
    )
MAX_FILE_SIZE = int(max_file_size_str)
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "bmp", "tiff", "tif"}


def get_file_type(filename: str) -> int:
    """
    Determine file type for PaddleOCR API.

    Args:
        filename: Name of the file

    Returns:
        0 for PDF, 1 for images
    """
    ext = filename.lower().split(".")[-1] if "." in filename else ""
    return 0 if ext == "pdf" else 1


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "service": "extract-paddle",
            "api_url": API_URL,
        }
    )


@app.route("/extract", methods=["POST"])
def extract_document():
    """
    Extract document using PaddleOCR API.

    Accepts:
        - multipart/form-data with 'file' field containing PDF or image

    Returns:
        - JSON with storage_data structure including extraction metadata
    """
    start_time = time.time()

    try:
        # Check if file is provided
        if "file" not in request.files:
            logger.error("No file provided in request")
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]

        if file.filename == "":
            logger.error("No file selected")
            return jsonify({"error": "No file selected"}), 400

        # Validate file type
        if not allowed_file(file.filename):
            logger.error(f"Invalid file type: {file.filename}")
            return (
                jsonify(
                    {
                        "error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
                    }
                ),
                400,
            )

        logger.info(f"Processing file: {file.filename}")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file.filename)[1]
        ) as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name

        try:
            # Get file size
            file_size = os.path.getsize(temp_path)
            if file_size > MAX_FILE_SIZE:
                logger.error(f"File too large: {file_size} bytes")
                return (
                    jsonify(
                        {
                            "error": f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
                        }
                    ),
                    400,
                )

            logger.info(f"File size: {file_size / 1024:.2f} KB")

            # Read and encode file
            read_start = time.time()
            with open(temp_path, "rb") as f:
                file_bytes = f.read()
                file_data = base64.b64encode(file_bytes).decode("ascii")
            read_time = time.time() - read_start
            logger.info(f"File read and encoded in {read_time:.2f} seconds")

            # Determine file type
            file_type = get_file_type(file.filename)
            logger.info(f"File type: {'PDF' if file_type == 0 else 'Image'}")

            # Prepare API request
            headers = {
                "Authorization": f"token {API_TOKEN}",
                "Content-Type": "application/json",
            }
            payload = {
                "file": file_data,
                "fileType": file_type,
                "useChartRecognition": True,
            }

            # Call PaddleOCR API
            logger.info(f"Sending request to API: {API_URL}")
            api_start = time.time()
            response = requests.post(
                API_URL, json=payload, headers=headers, timeout=600
            )
            api_time = time.time() - api_start
            logger.info(f"API call completed in {api_time:.2f} seconds")

            # Check response status
            if response.status_code != 200:
                logger.error(
                    f"API request failed with status code {response.status_code}"
                )
                return (
                    jsonify(
                        {
                            "error": "API request failed",
                            "status_code": response.status_code,
                            "message": response.text[:500],
                        }
                    ),
                    500,
                )

            # Parse response
            response_json = response.json()
            logger.info("API request successful")

            # Extract document ID from filename
            document_id = os.path.splitext(secure_filename(file.filename))[0]

            # Calculate total extraction time
            extraction_time = time.time() - start_time

            # Extract storage data
            logger.info("Extracting storage data...")
            storage_extract_start = time.time()
            storage_data = extract_storage_data(
                response_json,
                document_id=document_id,
                extraction_time_seconds=round(extraction_time, 2),
            )
            storage_extract_time = time.time() - storage_extract_start
            logger.info(f"Storage data extracted in {storage_extract_time:.2f} seconds")

            logger.info(
                f"Extraction complete: {len(storage_data['pages'])} pages, "
                f"{sum(len(page['sourceBlocks']) for page in storage_data['pages'])} total blocks"
            )

            # Include both processed storage_data and raw PaddleOCR response
            response_data = {
                **storage_data,
                "raw_response": response_json,  # Include raw, unconverted PaddleOCR response
            }

            return jsonify(response_data), 200

        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.debug(f"Cleaned up temporary file: {temp_path}")

    except requests.exceptions.Timeout:
        logger.error("API request timeout")
        return (
            jsonify(
                {"error": "API request timeout", "message": "Request took too long"}
            ),
            504,
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"API request error: {str(e)}")
        return (
            jsonify({"error": "API request failed", "message": str(e)}),
            500,
        )
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        extraction_time = time.time() - start_time
        return (
            jsonify(
                {
                    "error": "Internal server error",
                    "message": str(e),
                    "extractionTimeSeconds": round(extraction_time, 2),
                }
            ),
            500,
        )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5002))
    host = os.getenv("HOST", "0.0.0.0")
    debug = os.getenv("DEBUG", "False").lower() == "true"

    logger.info(f"Starting PaddleOCR extraction service on {host}:{port}")
    app.run(host=host, port=port, debug=debug)

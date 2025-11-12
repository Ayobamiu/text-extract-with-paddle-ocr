"""
Flask server for PaddleOCR document extraction.

Accepts PDF or image files and returns structured extraction data.
"""

import os
import base64
import time
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from utils.extractors import extract_storage_data
import requests

# Load environment variables from .env file
load_dotenv()

# Create logs directory if it doesn't exist
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Configure logging with both console and file handlers
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(log_format, date_format))

# File handler for all logs
log_file = LOGS_DIR / "app.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(log_format, date_format))

# File handler specifically for API errors
error_log_file = LOGS_DIR / "api_errors.log"
error_file_handler = logging.FileHandler(error_log_file)
error_file_handler.setLevel(logging.ERROR)
error_file_handler.setFormatter(logging.Formatter(log_format, date_format))

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)
root_logger.addHandler(error_file_handler)

# Get logger for this module
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
                "temperature": 0.2,
            }

            # Call PaddleOCR API with retry logic for rate limiting
            logger.info(f"Sending request to API: {API_URL}")
            api_start = time.time()

            # Retry configuration
            max_retries = 5
            base_delay = 2  # Base delay in seconds
            max_delay = 60  # Maximum delay in seconds

            response = None
            last_error = None

            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        API_URL, json=payload, headers=headers, timeout=600
                    )

                    # If successful, break out of retry loop
                    if response.status_code == 200:
                        break

                    # Handle 429 (Rate Limit) with retry
                    if response.status_code == 429:
                        retry_after = response.headers.get(
                            "Retry-After"
                        ) or response.headers.get("retry-after")

                        if retry_after:
                            try:
                                wait_time = int(retry_after)
                            except (ValueError, TypeError):
                                # If Retry-After is not a number, use exponential backoff
                                wait_time = min(base_delay * (2**attempt), max_delay)
                        else:
                            # No Retry-After header, use exponential backoff
                            wait_time = min(base_delay * (2**attempt), max_delay)

                        if attempt < max_retries - 1:
                            logger.warning(
                                f"Rate limit hit (429) on attempt {attempt + 1}/{max_retries}. "
                                f"Waiting {wait_time} seconds before retry..."
                            )
                            time.sleep(wait_time)
                            continue
                        else:
                            # Last attempt failed
                            logger.error(
                                f"Rate limit hit (429) after {max_retries} attempts. "
                                f"Giving up."
                            )
                    else:
                        # Non-429 error, don't retry
                        break

                except requests.exceptions.RequestException as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait_time = min(base_delay * (2**attempt), max_delay)
                        logger.warning(
                            f"Request exception on attempt {attempt + 1}/{max_retries}: {str(e)}. "
                            f"Waiting {wait_time} seconds before retry..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"Request failed after {max_retries} attempts: {str(e)}"
                        )

            api_time = time.time() - api_start
            logger.info(f"API call completed in {api_time:.2f} seconds")

            # Check response status
            if response is None or response.status_code != 200:
                error_details = {
                    "timestamp": datetime.now().isoformat(),
                    "filename": file.filename,
                    "file_size": file_size,
                    "status_code": response.status_code
                    if response
                    else "RequestException",
                    "api_url": API_URL,
                    "response_time_seconds": round(api_time, 2),
                    "response_headers": dict(response.headers) if response else None,
                    "response_text": (
                        response.text[:2000] if response else str(last_error)
                    )
                    if last_error or response
                    else "No response",
                    "retry_after": (
                        response.headers.get("Retry-After")
                        or response.headers.get("retry-after")
                        if response
                        else None
                    ),
                    "retries_attempted": max_retries
                    if response and response.status_code == 429
                    else 1,
                }

                # Log detailed error to file
                logger.error(
                    f"API request failed with status code {error_details['status_code']}\n"
                    f"Error details: {error_details}"
                )

                # Provide more specific error message for 429
                if response and response.status_code == 429:
                    error_message = (
                        "Rate limit exceeded. The API is temporarily limiting requests. "
                        "Please wait a few moments and try again, or process files at a slower rate."
                    )
                else:
                    error_message = (
                        response.text[:500]
                        if response
                        else str(last_error)
                        if last_error
                        else "Unknown error"
                    )

                return (
                    jsonify(
                        {
                            "error": "API request failed",
                            "status_code": error_details["status_code"],
                            "message": error_message,
                        }
                    ),
                    500 if response and response.status_code != 429 else 429,
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
        error_details = {
            "timestamp": datetime.now().isoformat(),
            "error_type": "Timeout",
            "api_url": API_URL,
            "timeout_seconds": 600,
        }
        logger.error(f"API request timeout\nError details: {error_details}")
        return (
            jsonify(
                {"error": "API request timeout", "message": "Request took too long"}
            ),
            504,
        )
    except requests.exceptions.RequestException as e:
        error_details = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(e).__name__,
            "error_message": str(e),
            "api_url": API_URL,
        }

        # If it's a response exception, include response details
        if hasattr(e, "response") and e.response is not None:
            error_details.update(
                {
                    "status_code": e.response.status_code,
                    "response_headers": dict(e.response.headers),
                    "response_text": e.response.text[:2000]
                    if e.response.text
                    else None,
                    "retry_after": e.response.headers.get("Retry-After")
                    or e.response.headers.get("retry-after"),
                }
            )

        logger.error(f"API request error: {str(e)}\nError details: {error_details}")

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

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
from pypdf import PdfReader, PdfWriter
from typing import List, Dict, Any

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
MAX_PAGES_PER_CHUNK = 40  # Paddle API limit


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


def split_pdf_into_chunks(
    pdf_path: str, chunk_size: int = MAX_PAGES_PER_CHUNK
) -> List[str]:
    """
    Split a PDF file into chunks of specified page size.

    Args:
        pdf_path: Path to the PDF file
        chunk_size: Maximum number of pages per chunk (default: 40)

    Returns:
        List of temporary file paths for each chunk
    """
    chunks = []
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    logger.info(
        f"Splitting PDF with {total_pages} pages into chunks of {chunk_size} pages"
    )

    if total_pages <= chunk_size:
        # No need to split
        return [pdf_path]

    chunk_count = (total_pages + chunk_size - 1) // chunk_size  # Ceiling division

    for chunk_idx in range(chunk_count):
        start_page = chunk_idx * chunk_size
        end_page = min(start_page + chunk_size, total_pages)

        # Create a new PDF writer for this chunk
        writer = PdfWriter()

        # Add pages to the chunk
        for page_num in range(start_page, end_page):
            writer.add_page(reader.pages[page_num])

        # Save chunk to temporary file
        chunk_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf", prefix=f"chunk_{chunk_idx}_"
        )
        writer.write(chunk_file)
        chunk_file.close()

        chunks.append(chunk_file.name)
        logger.info(
            f"Created chunk {chunk_idx + 1}/{chunk_count}: pages {start_page + 1}-{end_page} -> {chunk_file.name}"
        )

    return chunks


def merge_chunk_results(
    chunk_results: List[Dict[str, Any]],
    document_id: str,
    extraction_time_seconds: float,
) -> Dict[str, Any]:
    """
    Merge results from multiple PDF chunks into a single result.

    Args:
        chunk_results: List of storage_data dictionaries from each chunk
        document_id: Document identifier
        extraction_time_seconds: Total extraction time

    Returns:
        Merged storage_data dictionary
    """
    if not chunk_results:
        raise ValueError("No chunk results to merge")

    if len(chunk_results) == 1:
        # Only one chunk, return as-is (but update extraction time)
        result = chunk_results[0].copy()
        if "extractionMetadata" in result:
            result["extractionMetadata"]["extractionTimeSeconds"] = (
                extraction_time_seconds
            )
        return result

    logger.info(f"Merging {len(chunk_results)} chunk results")

    # Start with the first chunk's structure
    merged = chunk_results[0].copy()

    # Merge pages from all chunks, adjusting page indices
    all_pages = []
    all_raw_pages = []
    all_page_dimensions = []
    total_pages = 0

    for chunk_idx, chunk_result in enumerate(chunk_results):
        chunk_pages = chunk_result.get("pages", [])
        chunk_metadata = chunk_result.get("extractionMetadata", {})
        chunk_raw = chunk_result.get("raw_response", {})

        # Get pages from raw response for merging
        chunk_raw_pages = chunk_raw.get("result", {}).get("layoutParsingResults", [])

        # Get page dimensions
        chunk_page_dims = chunk_metadata.get("pages", [])

        # Adjust page indices to be sequential across chunks
        for page in chunk_pages:
            page["pageIndex"] = total_pages
            all_pages.append(page)
            total_pages += 1

        # Collect raw pages
        all_raw_pages.extend(chunk_raw_pages)

        # Collect page dimensions
        all_page_dimensions.extend(chunk_page_dims)

    # Update merged result
    merged["pages"] = all_pages

    # Update extraction metadata
    if "extractionMetadata" in merged:
        merged["extractionMetadata"]["numPages"] = total_pages
        merged["extractionMetadata"]["pages"] = all_page_dimensions
        merged["extractionMetadata"]["extractionTimeSeconds"] = extraction_time_seconds

    # Merge raw responses
    if "raw_response" in merged:
        # Get the structure from first chunk
        first_raw = chunk_results[0].get("raw_response", {})
        merged_raw = first_raw.copy()

        # Update layout parsing results
        if "result" in merged_raw:
            merged_raw["result"]["layoutParsingResults"] = all_raw_pages

            # Update dataInfo
            if "dataInfo" in merged_raw["result"]:
                merged_raw["result"]["dataInfo"]["numPages"] = total_pages
                merged_raw["result"]["dataInfo"]["pages"] = all_page_dimensions

        merged["raw_response"] = merged_raw

    logger.info(
        f"Merged result: {total_pages} total pages from {len(chunk_results)} chunks"
    )

    return merged


def process_single_chunk(
    chunk_path: str, file_type: int, headers: Dict[str, str]
) -> Dict[str, Any]:
    """
    Process a single PDF chunk through the PaddleOCR API.

    Args:
        chunk_path: Path to the PDF chunk file
        file_type: File type (0 for PDF, 1 for image)
        headers: HTTP headers for API request

    Returns:
        API response JSON
    """
    # Read and encode chunk file
    with open(chunk_path, "rb") as f:
        file_bytes = f.read()
        file_data = base64.b64encode(file_bytes).decode("ascii")

    required_payload = {
        "file": file_data,
        "fileType": file_type,
        "temperature": 0.2,
    }
    optional_payload = {
        "useDocUnwarping": True,
        "repetitionPenalty": 1.2,  # Helps reduce duplicate text
    }
    payload = {**required_payload, **optional_payload}

    # Call PaddleOCR API with retry logic
    max_retries = 5
    base_delay = 2
    max_delay = 60

    response = None
    last_error = None

    for attempt in range(max_retries):
        try:
            response = requests.post(
                API_URL, json=payload, headers=headers, timeout=600
            )

            if response.status_code == 200:
                break

            if response.status_code == 429:
                retry_after = response.headers.get(
                    "Retry-After"
                ) or response.headers.get("retry-after")

                if retry_after:
                    try:
                        wait_time = int(retry_after)
                    except (ValueError, TypeError):
                        wait_time = min(base_delay * (2**attempt), max_delay)
                else:
                    wait_time = min(base_delay * (2**attempt), max_delay)

                if attempt < max_retries - 1:
                    logger.warning(
                        f"Rate limit hit (429) on chunk processing attempt {attempt + 1}/{max_retries}. "
                        f"Waiting {wait_time} seconds before retry..."
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(
                        f"Rate limit hit (429) after {max_retries} attempts for chunk"
                    )
            else:
                break

        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = min(base_delay * (2**attempt), max_delay)
                logger.warning(
                    f"Request exception on chunk processing attempt {attempt + 1}/{max_retries}: {str(e)}. "
                    f"Waiting {wait_time} seconds before retry..."
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    f"Request failed after {max_retries} attempts for chunk: {str(e)}"
                )

    if response is None or response.status_code != 200:
        error_msg = (
            response.text[:500]
            if response
            else str(last_error)
            if last_error
            else "Unknown error"
        )
        raise Exception(f"API request failed for chunk: {error_msg}")

    return response.json()


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

            # Determine file type
            file_type = get_file_type(file.filename)
            logger.info(f"File type: {'PDF' if file_type == 0 else 'Image'}")

            # Prepare API request headers
            headers = {
                "Authorization": f"token {API_TOKEN}",
                "Content-Type": "application/json",
            }

            # Extract document ID from filename
            document_id = os.path.splitext(secure_filename(file.filename))[0]

            # Check if PDF needs chunking
            chunk_paths = []

            if file_type == 0:  # PDF
                try:
                    reader = PdfReader(temp_path)
                    total_pages = len(reader.pages)
                    logger.info(f"PDF has {total_pages} pages")

                    if total_pages > MAX_PAGES_PER_CHUNK:
                        needs_chunking = True
                        logger.info(
                            f"PDF exceeds {MAX_PAGES_PER_CHUNK} page limit, will split into chunks"
                        )
                        chunk_paths = split_pdf_into_chunks(temp_path)
                    else:
                        chunk_paths = [temp_path]
                except Exception as e:
                    logger.error(f"Error reading PDF: {str(e)}")
                    return (
                        jsonify(
                            {"error": "Failed to read PDF file", "message": str(e)}
                        ),
                        400,
                    )
            else:
                # Image file, no chunking needed
                chunk_paths = [temp_path]

            # Process chunks
            chunk_results = []
            chunk_temp_files = []  # Track chunk files to clean up

            try:
                for chunk_idx, chunk_path in enumerate(chunk_paths):
                    logger.info(
                        f"Processing chunk {chunk_idx + 1}/{len(chunk_paths)}: {chunk_path}"
                    )

                    # Track if this is a temporary chunk file (not the original)
                    is_temp_chunk = chunk_path != temp_path
                    if is_temp_chunk:
                        chunk_temp_files.append(chunk_path)

                    # Process chunk
                    api_start = time.time()
                    response_json = process_single_chunk(chunk_path, file_type, headers)
                    api_time = time.time() - api_start
                    logger.info(
                        f"Chunk {chunk_idx + 1} processed in {api_time:.2f} seconds"
                    )

                    # Extract storage data for this chunk
                    storage_data = extract_storage_data(
                        response_json,
                        document_id=document_id,
                        extraction_time_seconds=round(api_time, 2),
                    )

                    # Include raw response for merging
                    storage_data["raw_response"] = response_json
                    chunk_results.append(storage_data)

                # Merge results if multiple chunks
                if len(chunk_results) > 1:
                    logger.info("Merging chunk results...")
                    extraction_time = time.time() - start_time
                    merged_result = merge_chunk_results(
                        chunk_results,
                        document_id=document_id,
                        extraction_time_seconds=round(extraction_time, 2),
                    )
                    response_data = merged_result
                else:
                    # Single chunk, update extraction time
                    extraction_time = time.time() - start_time
                    chunk_results[0]["extractionMetadata"]["extractionTimeSeconds"] = (
                        round(extraction_time, 2)
                    )
                    response_data = chunk_results[0]

                logger.info(
                    f"Extraction complete: {len(response_data['pages'])} pages, "
                    f"{sum(len(page['sourceBlocks']) for page in response_data['pages'])} total blocks"
                )

                return jsonify(response_data), 200

            except Exception as chunk_error:
                logger.error(
                    f"Error processing chunks: {str(chunk_error)}", exc_info=True
                )
                return (
                    jsonify(
                        {
                            "error": "Error processing document chunks",
                            "message": str(chunk_error),
                        }
                    ),
                    500,
                )
            finally:
                # Clean up temporary chunk files
                for chunk_file in chunk_temp_files:
                    if os.path.exists(chunk_file):
                        try:
                            os.remove(chunk_file)
                            logger.debug(f"Cleaned up chunk file: {chunk_file}")
                        except Exception as e:
                            logger.warning(
                                f"Failed to clean up chunk file {chunk_file}: {e}"
                            )

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

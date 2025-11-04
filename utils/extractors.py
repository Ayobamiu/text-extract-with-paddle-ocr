"""
Data extraction functions for PaddleOCR full_response.json

Functions:
1. extract_storage_data() - Extracts processed data for storage
2. extract_openai_feed_data() - Extracts data to feed to OpenAI API
"""

import json
import re
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from .table_converter import html_to_markdown_table

logger = logging.getLogger(__name__)


def convert_html_tables_in_markdown(markdown_text: str) -> str:
    """
    Convert all HTML tables in markdown text to markdown format.

    This function finds all <table>...</table> blocks in the markdown text,
    converts each one to markdown format using html_to_markdown_table(),
    and replaces them in place while preserving the rest of the text.

    Args:
        markdown_text: Markdown string that may contain HTML tables

    Returns:
        Markdown string with all HTML tables converted to markdown format
    """
    if not markdown_text:
        return markdown_text

    # Quick check: if no tables, return immediately (performance optimization)
    if "<table" not in markdown_text.lower():
        return markdown_text

    # Pattern to match complete <table>...</table> blocks
    # Using non-greedy matching and DOTALL to handle multi-line tables
    # Pre-compile for better performance
    # Pattern matches: <table> with optional attributes (single or double quotes, spaces, etc.)
    # Handles: attributes with spaces, single/double quotes, newlines, case-insensitive
    # More explicit pattern to catch edge cases
    table_pattern = re.compile(r"<table\b[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE)

    def replace_table(match):
        """Replace a single HTML table with its markdown equivalent."""
        html_table = match.group(0)
        try:
            # html_to_markdown_table already handles finding and converting tables
            # It will extract the table content and convert it to markdown
            markdown_table = html_to_markdown_table(html_table)
            return markdown_table if markdown_table else html_table
        except Exception as e:
            # If conversion fails, keep original HTML and log warning
            print(f"Warning: Failed to convert table in markdown: {e}")
            return html_table

    # Find and replace all tables in the markdown text
    converted_markdown = table_pattern.sub(replace_table, markdown_text)

    return converted_markdown


def extract_storage_data(
    full_response: Dict[str, Any],
    document_id: Optional[str] = None,
    extraction_time_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Extract and transform data from full_response.json for storage.

    This function processes the raw API response into an optimized structure
    that includes source blocks with coordinates, markdown, and mappings
    needed for PDF-markdown linking.

    Args:
        full_response: The full response JSON from PaddleOCR API
        document_id: Optional document identifier (extracted from logId if not provided)
        extraction_time_seconds: Optional extraction time in seconds to include in metadata

    Returns:
        Dictionary containing processed extraction data ready for storage
    """
    result = full_response.get("result", {})
    layout_results = result.get("layoutParsingResults", [])
    data_info = result.get("dataInfo", {})

    # Extract document ID from logId if not provided
    if not document_id:
        document_id = full_response.get("logId", "unknown")

    # Build pages array with processed data
    pages = []

    for page_index, page_data in enumerate(layout_results):
        pruned_result = page_data.get("prunedResult", {})
        parsing_res_list = pruned_result.get("parsing_res_list", [])
        layout_det_res = pruned_result.get("layout_det_res", {})
        markdown_data = page_data.get("markdown", {})
        output_images = page_data.get("outputImages", {})

        # Get page dimensions from dataInfo
        page_dimensions = data_info.get("pages", [])
        page_width = (
            page_dimensions[page_index]["width"]
            if page_index < len(page_dimensions)
            else None
        )
        page_height = (
            page_dimensions[page_index]["height"]
            if page_index < len(page_dimensions)
            else None
        )

        # Process source blocks with markdown mapping
        source_blocks = []
        markdown_text = markdown_data.get("text", "")
        markdown_images = markdown_data.get("images", {})

        # Download images from PaddleOCR and upload to S3, then replace URLs
        image_url_mapping = {}  # Maps image_name -> new S3 URL
        if markdown_images:
            try:
                from .s3_helper import (
                    upload_image_to_s3,
                    get_content_type_from_filename,
                )
                import requests

                for image_name, paddleocr_url in markdown_images.items():
                    if not image_name or not paddleocr_url:
                        continue

                    try:
                        # Download image from PaddleOCR
                        logger.info(
                            f"📥 Downloading image: {image_name} from PaddleOCR"
                        )
                        response = requests.get(paddleocr_url, timeout=30, stream=True)
                        response.raise_for_status()

                        image_data = response.content

                        # Upload to S3
                        s3_key = f"images/{image_name}"
                        content_type = get_content_type_from_filename(image_name)
                        s3_url = upload_image_to_s3(
                            image_data, s3_key, content_type=content_type
                        )

                        if s3_url:
                            image_url_mapping[image_name] = s3_url
                            logger.info(
                                f"✅ Uploaded image to S3: {image_name} -> {s3_url[:50]}..."
                            )
                        else:
                            # Fallback to original PaddleOCR URL if S3 upload fails
                            image_url_mapping[image_name] = paddleocr_url
                            logger.warning(
                                f"⚠️ S3 upload failed for {image_name}, using original URL"
                            )

                    except Exception as img_error:
                        # Fallback to original URL on error
                        logger.error(
                            f"❌ Error processing image {image_name}: {img_error}"
                        )
                        image_url_mapping[image_name] = paddleocr_url

            except ImportError:
                # S3 helper not available, use original URLs
                logger.warning("S3 helper not available, using original PaddleOCR URLs")
                image_url_mapping = markdown_images
            except Exception as e:
                logger.error(f"Error initializing S3 upload: {e}")
                image_url_mapping = markdown_images

        # Replace image keys in src attributes with actual URLs (S3 or PaddleOCR)
        # Pattern: src="imageName" or src='imageName' -> src="imgURL" or src='imgURL'
        for image_name, image_url in image_url_mapping.items():
            if image_name and image_url:
                # Escape special regex characters in image_name
                escaped_image_name = re.escape(image_name)
                # Replace src="imageName" or src='imageName' with src="imgURL"
                # We'll use double quotes for consistency
                markdown_text = re.sub(
                    rf'src=["\']{escaped_image_name}["\']',
                    f'src="{image_url}"',
                    markdown_text,
                )

        # Keep HTML tables as HTML (don't convert to markdown)
        # Note: HTML tables will remain in their original format

        current_markdown_offset = 0

        for block in parsing_res_list:
            block_id = str(block.get("block_id", ""))
            block_label = block.get("block_label", "")
            block_content = block.get("block_content", "")
            block_bbox = block.get("block_bbox", [])
            block_order = block.get("block_order")

            # Keep HTML tables as HTML (don't convert to markdown)
            # Replace escaped quotes with single quotes in table content
            if block_label and block_label.lower() == "table" and block_content:
                # Replace \" with ' in table content
                # Using raw string or double backslash to match escaped quote
                block_content = block_content.replace('\\"', "'")

            # Try to find this block's content in the markdown
            # This is a simplified approach - in production, you might need
            # more sophisticated matching based on block_order and content
            markdown_index = None
            markdown_length = len(block_content) if block_content else 0

            # Search for block content in markdown (approximate matching)
            if block_content and markdown_text:
                # Try exact match first
                search_start = max(0, current_markdown_offset - 100)
                search_text = markdown_text[
                    search_start : search_start + len(markdown_text) * 2
                ]
                content_pos = search_text.find(block_content)

                if content_pos >= 0:
                    markdown_index = search_start + content_pos
                    current_markdown_offset = markdown_index + markdown_length
                else:
                    # Try finding a substring (for blocks that might be formatted differently in markdown)
                    block_words = block_content.split()[:3]  # First 3 words
                    if block_words:
                        search_pattern = " ".join(block_words)
                        content_pos = markdown_text.find(search_pattern, search_start)
                        if content_pos >= 0:
                            markdown_index = content_pos
                            current_markdown_offset = markdown_index + len(
                                search_pattern
                            )

            source_blocks.append(
                {
                    "blockId": block_id,
                    "blockLabel": block_label,
                    "blockContent": block_content,
                    "blockBbox": block_bbox,
                    "blockOrder": block_order,
                    "markdownIndex": markdown_index,
                    "markdownLength": markdown_length,
                    "markdownOffset": markdown_index
                    if markdown_index is not None
                    else current_markdown_offset,
                }
            )

        # Process layout detection boxes (for visualization)
        layout_boxes = []
        for box in layout_det_res.get("boxes", []):
            layout_boxes.append(
                {
                    "clsId": box.get("cls_id"),
                    "label": box.get("label"),
                    "score": box.get("score"),
                    "coordinate": box.get("coordinate", []),
                }
            )

        pages.append(
            {
                "pageIndex": page_index,
                "pageWidth": page_width,
                "pageHeight": page_height,
                "sourceBlocks": source_blocks,
                "markdown": {
                    "text": markdown_text,
                    "images": markdown_data.get("images", {}),
                },
                "layoutBoxes": layout_boxes,
                "outputImages": output_images,
                "inputImage": page_data.get("inputImage"),
            }
        )

    # Build extraction metadata
    extraction_metadata = {
        "logId": full_response.get("logId"),
        "timestamp": datetime.utcnow().isoformat(),
        "fileType": data_info.get("type", "pdf"),
        "numPages": data_info.get("numPages", len(pages)),
        "pages": data_info.get("pages", []),
        "extractionTimeSeconds": extraction_time_seconds,
    }

    return {
        "documentId": document_id,
        "extractionMetadata": extraction_metadata,
        "pages": pages,
    }


def extract_openai_feed_from_storage(
    storage_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract OpenAI feed data from storage_data.

    Simplified format to reduce token usage - only includes essential data:
    - pageIndex for each page
    - blockId and blockContent for source blocks (no markdown, bboxes, or labels)

    This is the preferred method since storage_data is the single source of truth.

    Args:
        storage_data: The processed storage data from extract_storage_data()

    Returns:
        Dictionary containing simplified page and source blocks ready for OpenAI API
    """
    pages_data = storage_data.get("pages", [])

    pages = []
    for page in pages_data:
        # Extract only blocks with content (simplified format)
        source_blocks = []
        for block in page.get("sourceBlocks", []):
            block_content = block.get("blockContent", "").strip()
            if not block_content:
                continue

            source_blocks.append(
                {
                    "blockId": block.get("blockId"),
                    "blockContent": block_content,
                }
            )

        pages.append(
            {
                "pageIndex": page.get("pageIndex"),
                "sourceBlocks": source_blocks,
            }
        )

    return {"pages": pages}


def _format_block_content_by_label(content: str, block_label: str) -> str:
    """
    Format block content based on its label using markdown syntax.

    Args:
        content: Block text content
        block_label: Block label/type (e.g., "header", "footer", "doc_title")

    Returns:
        Formatted markdown string
    """
    if not content:
        return ""

    label_lower = block_label.lower() if block_label else ""

    # Apply markdown formatting based on label
    if label_lower == "doc_title":
        # Document title - use h1
        return f"# {content}"
    elif label_lower == "header":
        # Header - use h2
        return f"## {content}"
    elif label_lower == "paragraph_title":
        # Paragraph/section title - use h3
        return f"### {content}"
    elif label_lower == "figure_title":
        # Figure caption - use h4 or italic
        return f"#### {content}"
    elif label_lower == "footer":
        # Footer - use italic or small text
        return f"*{content}*"
    elif label_lower == "vision_footnote":
        # Footnote - use smaller text
        return f"<small>{content}</small>"
    elif label_lower == "table":
        # Table - content is already converted to markdown in extract_storage_data()
        # Return as-is (with proper spacing for markdown tables)
        return content
    elif label_lower == "number":
        # Numbers - often formatting or emphasis
        return f"**{content}**"
    else:
        # Default: "text" or unknown - return as plain text
        return content


def extract_openai_feed_markdown_from_storage(
    storage_data: Dict[str, Any], blocked: bool = False
) -> str:
    """
    Extract OpenAI feed data from storage_data in markdown format.

    Converts the structured storage_data into a markdown text format that is
    more token-efficient than JSON while still preserving page and block identifiers
    for source location tracking. Applies markdown formatting based on block labels.

    Format:
    === PAGE 1 ===
    [BLOCK: 0]
    # Document Title (formatted based on label)

    [BLOCK: 1]
    ## Header Text (formatted based on label)

    === PAGE 2 ===
    ...

    Args:
        storage_data: The processed storage data from extract_storage_data()

    Returns:
        Markdown string ready for OpenAI API consumption
    """
    pages_data = storage_data.get("pages", [])

    lines = []
    for page in pages_data:
        page_index = page.get("pageIndex", 0)
        source_blocks = page.get("sourceBlocks", [])

        if blocked:
            # Page header (display page number starting from 1 instead of 0 for blocked mode)
            lines.append(f"=== PAGE {page_index + 1} ===\n")

        # Add each block with content
        for block in source_blocks:
            block_id = block.get("blockId", "")
            block_content = block.get("blockContent", "").strip()
            block_label = block.get("blockLabel", "")

            if not block_content:
                continue

            # Block identifier
            if blocked:
                lines.append(f"[BLOCK: {block_id}]\n")

            # Format content based on block label
            formatted_content = _format_block_content_by_label(
                block_content, block_label
            )
            lines.append(formatted_content)

            # Empty line between blocks for readability
            lines.append("")

    return "\n".join(lines)


def convert_to_mineru_format(
    storage_data: Dict[str, Any],
    filename: Optional[str] = None,
    file_size_mb: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Convert PaddleOCR storage_data to MinerU-compatible format.

    This function transforms the structured PaddleOCR output into a format
    compatible with MinerU extractor responses, making it easy to integrate
    PaddleOCR results with existing MinerU-based workflows.

    Args:
        storage_data: The processed storage data from extract_storage_data()
        filename: Original filename (optional, extracted from documentId if not provided)
        file_size_mb: File size in MB (optional)

    Returns:
        Dictionary in MinerU-compatible format:
        {
            "success": True,
            "filename": "...",
            "data": {
                "markdown": "...",
                "metadata": {...},
                "pages": [...],
                "tables": [...],
                "full_text": "...",
                "raw_text": "...",
                "structured_data": {...}
            }
        }
    """
    pages_data = storage_data.get("pages", [])
    document_id = storage_data.get("documentId", "unknown")
    extraction_metadata = storage_data.get("extractionMetadata", {})

    # Use provided filename or extract from documentId
    if not filename:
        filename = document_id

    # Combine markdown from all pages
    markdown_parts = []
    all_text_parts = []
    tables = []
    structured_pages = []

    for page in pages_data:
        page_index = page.get("pageIndex", 0)
        page_markdown = page.get("markdown", {})
        markdown_text = (
            page_markdown.get("text", "")
            if isinstance(page_markdown, dict)
            else str(page_markdown)
        )

        markdown_parts.append(markdown_text)

        # Extract text from source blocks (fallback if no markdown)
        if not markdown_text:
            page_blocks = page.get("sourceBlocks", [])
            block_texts = [
                block.get("blockContent", "")
                for block in page_blocks
                if block.get("blockContent", "").strip()
            ]
            markdown_text = "\n".join(block_texts)

        all_text_parts.append(markdown_text)

        # Extract tables from source blocks with table label
        page_blocks = page.get("sourceBlocks", [])
        for block in page_blocks:
            block_label = block.get("blockLabel", "").lower()
            if "table" in block_label:
                table_content = block.get("blockContent", "")
                if table_content.strip():
                    tables.append(
                        {
                            "table_id": len(tables) + 1,
                            "page": page_index
                            + 1,  # 1-indexed for MinerU compatibility
                            "data": table_content,
                            "bbox": block.get("blockBbox", []),
                            "block_id": block.get("blockId"),
                        }
                    )

        # Build structured page data
        structured_pages.append(
            {
                "page_number": page_index + 1,  # 1-indexed
                "text": markdown_text,
                "markdown": markdown_text,
                "source_blocks": [
                    {
                        "blockId": block.get("blockId"),
                        "blockLabel": block.get("blockLabel"),
                        "blockContent": block.get("blockContent", ""),
                        "bbox": block.get("blockBbox", []),
                    }
                    for block in page_blocks
                ],
            }
        )

    # Combine all markdown/text
    full_markdown = "\n\n".join(markdown_parts)
    full_text = "\n\n".join(all_text_parts)

    # Build metadata
    num_pages = extraction_metadata.get("numPages", len(pages_data))
    extraction_time = extraction_metadata.get("extractionTimeSeconds")

    metadata = {
        "processing_method": "paddleocr",
        "total_pages": num_pages,
        "word_count": len(full_markdown.split()) if full_markdown else 0,
        "original_file": filename,
        "file_size_mb": file_size_mb,
        "extraction_time_seconds": extraction_time,
        "log_id": extraction_metadata.get("logId"),
        "file_type": extraction_metadata.get("fileType", "pdf"),
    }

    # Build structured data from source blocks
    structured_data = {
        "total_blocks": sum(len(page.get("sourceBlocks", [])) for page in pages_data),
        "block_labels": list(
            set(
                block.get("blockLabel", "")
                for page in pages_data
                for block in page.get("sourceBlocks", [])
                if block.get("blockLabel")
            )
        ),
        "layout_boxes": [
            {
                "pageIndex": page.get("pageIndex"),
                "boxes": page.get("layoutBoxes", []),
            }
            for page in pages_data
            if page.get("layoutBoxes")
        ],
    }

    return {
        "success": True,
        "filename": filename,
        "data": {
            "markdown": full_markdown,
            "metadata": metadata,
            "pages": structured_pages,
            "tables": tables,
            "full_text": full_text,
            "raw_text": full_text,  # Same as full_text for PaddleOCR
            "structured_data": structured_data,
        },
    }


def extract_openai_feed_data(
    full_response: Dict[str, Any], document_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract data from full_response.json to feed to OpenAI API.

    DEPRECATED: Use extract_openai_feed_from_storage() instead.
    This function is kept for backward compatibility.

    This function extracts markdown text and source blocks with coordinates,
    formatted for OpenAI API consumption with schema extraction in mind.

    Args:
        full_response: The full response JSON from PaddleOCR API
        document_id: Optional document identifier

    Returns:
        Dictionary containing markdown and source blocks ready for OpenAI API
    """
    result = full_response.get("result", {})
    layout_results = result.get("layoutParsingResults", [])
    data_info = result.get("dataInfo", {})

    # Extract document ID
    if not document_id:
        document_id = full_response.get("logId", "unknown")

    # Get page dimensions
    page_dimensions = data_info.get("pages", [])

    # Build pages array with markdown and source blocks
    pages = []

    for page_index, page_data in enumerate(layout_results):
        pruned_result = page_data.get("prunedResult", {})
        parsing_res_list = pruned_result.get("parsing_res_list", [])
        markdown_data = page_data.get("markdown", {})

        # Get page dimensions
        page_width = (
            page_dimensions[page_index]["width"]
            if page_index < len(page_dimensions)
            else None
        )
        page_height = (
            page_dimensions[page_index]["height"]
            if page_index < len(page_dimensions)
            else None
        )

        # Extract markdown text
        markdown_text = markdown_data.get("text", "")

        # Extract source blocks (only relevant data for AI)
        source_blocks = []
        for block in parsing_res_list:
            # Only include blocks that have content
            block_content = block.get("block_content", "").strip()
            if not block_content:
                continue

            source_blocks.append(
                {
                    "blockId": str(block.get("block_id", "")),
                    "blockLabel": block.get("block_label", ""),
                    "blockContent": block_content,
                    "blockBbox": block.get("block_bbox", []),
                }
            )

        pages.append(
            {
                "pageIndex": page_index,
                "pageWidth": page_width,
                "pageHeight": page_height,
                "markdown": markdown_text,
                "sourceBlocks": source_blocks,
            }
        )

    return {"documentId": document_id, "numPages": len(pages), "pages": pages}


def build_markdown_to_block_map(
    storage_data: Dict[str, Any],
) -> Dict[int, Dict[str, Any]]:
    """
    Build a mapping from markdown character positions to source blocks.

    This mapping enables hover functionality where hovering on markdown text
    highlights the corresponding location in the PDF.

    Args:
        storage_data: The processed storage data from extract_storage_data()

    Returns:
        Dictionary mapping character positions to block information
        Key: character offset in markdown
        Value: { blockId, bbox, pageIndex }
    """
    markdown_map = {}

    for page in storage_data.get("pages", []):
        page_index = page.get("pageIndex")
        markdown_text = page.get("markdown", {}).get("text", "")

        for block in page.get("sourceBlocks", []):
            markdown_index = block.get("markdownIndex")
            markdown_length = block.get("markdownLength", 0)

            if markdown_index is not None:
                # Map each character in the range to this block
                for char_offset in range(
                    markdown_index, markdown_index + markdown_length
                ):
                    if char_offset < len(markdown_text):
                        markdown_map[char_offset] = {
                            "blockId": block.get("blockId"),
                            "bbox": block.get("blockBbox"),
                            "pageIndex": page_index,
                            "blockLabel": block.get("blockLabel"),
                        }

    return markdown_map


# Example usage and testing
if __name__ == "__main__":
    # Load the full response
    with open("output-2/full_response.json", "r", encoding="utf-8") as f:
        full_response = json.load(f)

    # Extract storage data
    print("Extracting storage data...")
    storage_data = extract_storage_data(full_response, document_id="07126_WF")
    print(f"✓ Extracted {len(storage_data['pages'])} pages")
    print(f"  Document ID: {storage_data['documentId']}")
    print(
        f"  First page has {len(storage_data['pages'][0]['sourceBlocks'])} source blocks"
    )

    # Extract OpenAI feed data from storage_data (preferred method)
    print("\nExtracting OpenAI feed data from storage_data...")
    openai_data = extract_openai_feed_from_storage(storage_data)
    print(f"✓ Extracted {len(openai_data['pages'])} pages for OpenAI")
    print(
        f"  First page markdown length: {len(openai_data['pages'][0]['markdown'])} characters"
    )
    print(
        f"  First page has {len(openai_data['pages'][0]['sourceBlocks'])} source blocks"
    )

    # Build markdown mapping
    print("\nBuilding markdown-to-block map...")
    markdown_map = build_markdown_to_block_map(storage_data)
    print(f"✓ Created mapping for {len(markdown_map)} character positions")

    # Save outputs for inspection
    with open("output-2/storage_data.json", "w", encoding="utf-8") as f:
        json.dump(storage_data, f, indent=2, ensure_ascii=False)
    print("\n✓ Saved storage_data.json")

    with open("output-2/openai_feed_data.json", "w", encoding="utf-8") as f:
        json.dump(openai_data, f, indent=2, ensure_ascii=False)
    print("✓ Saved openai_feed_data.json")

    with open("output-2/markdown_map.json", "w", encoding="utf-8") as f:
        json.dump(markdown_map, f, indent=2, ensure_ascii=False)
    print("✓ Saved markdown_map.json")

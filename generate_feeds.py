#!/usr/bin/env python3
"""
Simple script to generate storage_data.json and OpenAI feed markdown files.

Accepts either:
- full_response.json (from PaddleOCR API) - will regenerate storage_data.json
- storage_data.json (from /extract endpoint) - will use existing storage_data

Usage:
    python generate_feeds.py [input_file.json]

    If no file is provided, defaults to storage_data.json (or full_response.json if that exists)
"""

import json
import sys
from pathlib import Path
from utils.extractors import (
    extract_storage_data,
    extract_openai_feed_markdown_from_storage,
)

# Default input and output file paths
STORAGE_DATA_FILE = Path("storage_data.json")
FULL_RESPONSE_FILE = Path("full_response.json")
BLOCKED_FEED_FILE = Path("openai-feed-blocked.md")
UNBLOCKED_FEED_FILE = Path("openai-feed-unblocked.md")


def main():
    """Generate storage_data and markdown feeds."""

    # Get input file from command line or use default
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])
    else:
        # Try storage_data.json first, then full_response.json
        if STORAGE_DATA_FILE.exists():
            input_file = STORAGE_DATA_FILE
        elif FULL_RESPONSE_FILE.exists():
            input_file = FULL_RESPONSE_FILE
        else:
            print(
                f"❌ Error: Neither {STORAGE_DATA_FILE} nor {FULL_RESPONSE_FILE} found!"
            )
            print(f"   Current directory: {Path.cwd()}")
            print(
                "   Usage: python generate_feeds.py [full_response.json|storage_data.json]"
            )
            sys.exit(1)

    # Check if input file exists
    if not input_file.exists():
        print(f"❌ Error: {input_file} not found!")
        print(f"   Current directory: {Path.cwd()}")
        print(
            "   Usage: python generate_feeds.py [full_response.json|storage_data.json]"
        )
        sys.exit(1)

    print(f"📄 Loading {input_file}...")
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            input_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading {input_file}: {e}")
        sys.exit(1)

    # Determine if it's full_response or storage_data format
    is_full_response = isinstance(input_data, dict) and "parsing_res_list" in input_data
    is_storage_data = isinstance(input_data, dict) and "pages" in input_data

    if not (is_full_response or is_storage_data):
        print(f"❌ Error: {input_file} doesn't appear to be a valid format!")
        print(
            "   Expected either 'parsing_res_list' (full_response) or 'pages' (storage_data)"
        )
        sys.exit(1)

    # Extract storage_data from full_response if needed
    if is_full_response:
        print("   Detected: full_response.json format")
        print("   Regenerating storage_data.json (with table conversion)...")
        try:
            # Extract document_id from full_response or filename
            document_id = None
            # Try to get from log_id or other metadata in full_response
            if isinstance(input_data, dict):
                # Check common paths for document identifier
                if "log_id" in input_data:
                    document_id = input_data["log_id"][:10]  # Use first part of log_id
                elif "document_id" in input_data:
                    document_id = input_data["document_id"]

            # Fallback to filename-based extraction
            if not document_id:
                # Try to extract from filename (e.g., "full_response_06877.json" -> "06877")
                stem = input_file.stem
                parts = stem.replace("full_response", "").replace("_", "").strip()
                if parts:
                    document_id = parts
                else:
                    document_id = "unknown"

            storage_data = extract_storage_data(input_data, document_id=document_id)

            # Save regenerated storage_data
            with open(STORAGE_DATA_FILE, "w", encoding="utf-8") as f:
                json.dump(storage_data, f, indent=2, ensure_ascii=False)
            print(f"   ✅ Regenerated and saved: {STORAGE_DATA_FILE}")
        except Exception as e:
            print(f"❌ Error regenerating storage_data: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)
    else:
        print("   Detected: storage_data.json format")
        storage_data = input_data

        # If input was storage_data.json but it's outdated (contains HTML tables),
        # check if full_response.json exists and regenerate
        has_html_tables = any(
            "<table" in block.get("blockContent", "").lower()
            for page in storage_data.get("pages", [])
            for block in page.get("sourceBlocks", [])
            if block.get("blockLabel", "").lower() == "table"
        )

        if has_html_tables and FULL_RESPONSE_FILE.exists():
            print("   ⚠️  Warning: Found HTML tables in storage_data.json")
            print(f"   Found {FULL_RESPONSE_FILE} - regenerating storage_data.json...")
            try:
                with open(FULL_RESPONSE_FILE, "r", encoding="utf-8") as f:
                    full_response = json.load(f)
                document_id = storage_data.get("documentId", "unknown")
                storage_data = extract_storage_data(
                    full_response, document_id=document_id
                )

                with open(STORAGE_DATA_FILE, "w", encoding="utf-8") as f:
                    json.dump(storage_data, f, indent=2, ensure_ascii=False)
                print("   ✅ Regenerated storage_data.json with table conversion")
            except Exception as e:
                print(f"   ⚠️  Warning: Could not regenerate from full_response: {e}")
                print("   Continuing with existing storage_data.json...")

    document_id = storage_data.get("documentId", "unknown")
    num_pages = len(storage_data.get("pages", []))
    print(f"\n   Document ID: {document_id}")
    print(f"   Pages: {num_pages}")

    # Display statistics
    total_blocks = sum(
        len(page.get("sourceBlocks", [])) for page in storage_data.get("pages", [])
    )
    table_count = sum(
        1
        for page in storage_data.get("pages", [])
        for block in page.get("sourceBlocks", [])
        if block.get("blockLabel", "").lower() == "table"
    )

    print("\n📊 Document statistics:")
    print(f"   - Pages: {num_pages}")
    print(f"   - Total blocks: {total_blocks}")
    if table_count > 0:
        print(f"   - Tables: {table_count}")

    # Generate blocked markdown
    print("\n📝 Generating blocked OpenAI feed markdown...")
    try:
        blocked_markdown = extract_openai_feed_markdown_from_storage(
            storage_data, blocked=True
        )

        with open(BLOCKED_FEED_FILE, "w", encoding="utf-8") as f:
            f.write(blocked_markdown)

        print(f"   ✅ Saved: {BLOCKED_FEED_FILE}")
        print(f"   - Length: {len(blocked_markdown):,} characters")
    except Exception as e:
        print(f"❌ Error generating blocked feed: {e}")
        sys.exit(1)

    # Generate unblocked markdown
    print("\n📝 Generating unblocked OpenAI feed markdown...")
    try:
        unblocked_markdown = extract_openai_feed_markdown_from_storage(
            storage_data, blocked=False
        )

        with open(UNBLOCKED_FEED_FILE, "w", encoding="utf-8") as f:
            f.write(unblocked_markdown)

        print(f"   ✅ Saved: {UNBLOCKED_FEED_FILE}")
        print(f"   - Length: {len(unblocked_markdown):,} characters")
    except Exception as e:
        print(f"❌ Error generating unblocked feed: {e}")
        sys.exit(1)

    print("\n✅ All files generated successfully!")
    print("\n📁 Output files:")
    if is_full_response:
        print(f"   - {STORAGE_DATA_FILE} (regenerated)")
    print(f"   - {BLOCKED_FEED_FILE}")
    print(f"   - {UNBLOCKED_FEED_FILE}")


if __name__ == "__main__":
    main()

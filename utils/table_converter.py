"""
HTML table to Markdown converter.

Converts HTML table markup to markdown table format, supporting:
- Simple tables
- Complex tables with colspan and rowspan
- Multiple tables (concatenated)
"""

import re
from typing import List, Tuple, Optional


def html_to_markdown_table(html: str) -> str:
    """
    Convert HTML table(s) to markdown format.

    If multiple tables are present, they will be concatenated with blank lines.
    Supports colspan and rowspan attributes.

    Args:
        html: HTML string containing one or more <table> elements

    Returns:
        Markdown formatted table string (or concatenated tables if multiple)
    """
    # Find all table elements in the HTML
    table_pattern = r"<table[^>]*>(.*?)</table>"
    tables = re.findall(table_pattern, html, re.DOTALL | re.IGNORECASE)

    if not tables:
        return ""

    markdown_tables = []
    for table_html in tables:
        markdown = _convert_single_table(table_html)
        if markdown:
            markdown_tables.append(markdown)

    # Concatenate multiple tables with blank lines
    return "\n\n".join(markdown_tables)


def _convert_single_table(table_html: str) -> str:
    """
    Convert a single HTML table to markdown format.

    Args:
        table_html: HTML content inside <table> tags

    Returns:
        Markdown formatted table string
    """
    # Parse table structure
    rows = _parse_table_rows(table_html)

    if not rows:
        return ""

    # Build table grid to handle colspan/rowspan
    grid = _build_table_grid(rows)

    if not grid:
        return ""

    # Convert grid to markdown
    return _grid_to_markdown(grid)


def _parse_table_rows(table_html: str) -> List[List[dict]]:
    """
    Parse HTML table rows into a structured format.

    Returns:
        List of rows, each row is a list of cell dictionaries with:
        - content: cell text content
        - colspan: colspan attribute (default 1)
        - rowspan: rowspan attribute (default 1)
        - is_header: True if <th>, False if <td>
    """
    rows = []

    # Find all <tr> elements
    tr_pattern = r"<tr[^>]*>(.*?)</tr>"
    tr_matches = re.findall(tr_pattern, table_html, re.DOTALL | re.IGNORECASE)

    for tr_html in tr_matches:
        row_cells = []

        # Find all <th> and <td> elements
        cell_pattern = r"<(th|td)[^>]*>(.*?)</(?:th|td)>"
        cell_matches = re.findall(cell_pattern, tr_html, re.DOTALL | re.IGNORECASE)

        for tag, cell_html in cell_matches:
            # Extract attributes
            colspan = _extract_attribute(cell_html, "colspan", 1)
            rowspan = _extract_attribute(cell_html, "rowspan", 1)

            # Extract text content (remove nested tags for now, keep text)
            content = _extract_text_content(cell_html)

            row_cells.append(
                {
                    "content": content,
                    "colspan": colspan,
                    "rowspan": rowspan,
                    "is_header": (tag.lower() == "th"),
                }
            )

        if row_cells:
            rows.append(row_cells)

    return rows


def _extract_attribute(html: str, attr_name: str, default: int = 1) -> int:
    """Extract integer attribute value from HTML tag."""
    pattern = rf'{attr_name}\s*=\s*["\']?(\d+)["\']?'
    match = re.search(pattern, html, re.IGNORECASE)
    return int(match.group(1)) if match else default


def _extract_text_content(html: str) -> str:
    """Extract and clean text content from HTML cell."""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", html)
    # Clean whitespace
    text = " ".join(text.split())
    # Replace common HTML entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    return text.strip()


def _build_table_grid(rows: List[List[dict]]) -> List[List[Optional[str]]]:
    """
    Build a 2D grid representation of the table handling colspan and rowspan.

    Returns:
        2D list where each cell contains text content or None (if occupied by rowspan)
    """
    # First pass: determine grid dimensions
    max_cols = 0
    for row in rows:
        cols = sum(cell["colspan"] for cell in row)
        max_cols = max(max_cols, cols)

    # Initialize grid
    grid: List[List[Optional[str]]] = []
    # Track active rowspans: dict mapping (row_idx, col_idx) -> (content, colspan, remaining_rows)
    active_rowspans: dict = {}

    # Build grid row by row
    for row_idx, row in enumerate(rows):
        # Initialize new grid row
        grid_row: List[Optional[str]] = [None] * max_cols

        # First, apply rowspans from previous rows
        rowspans_to_remove = []
        for (orig_row, orig_col), (
            content,
            colspan,
            remaining,
        ) in active_rowspans.items():
            if remaining > 0:
                # Place rowspan content in current row
                grid_row[orig_col] = content
                # Mark colspan cells as empty string
                for c in range(1, colspan):
                    if orig_col + c < max_cols:
                        grid_row[orig_col + c] = ""

                # Update remaining rows
                active_rowspans[(orig_row, orig_col)] = (
                    content,
                    colspan,
                    remaining - 1,
                )
                if remaining - 1 == 0:
                    rowspans_to_remove.append((orig_row, orig_col))

        # Remove completed rowspans
        for key in rowspans_to_remove:
            del active_rowspans[key]

        # Now place cells from current row
        col_idx = 0
        for cell in row:
            # Find next available column (skip rowspan occupied cells)
            while col_idx < max_cols and grid_row[col_idx] is not None:
                col_idx += 1

            if col_idx >= max_cols:
                break

            # Place current cell
            grid_row[col_idx] = cell["content"]

            # Handle colspan: mark spanned columns as empty string
            for c in range(1, cell["colspan"]):
                if col_idx + c < max_cols:
                    grid_row[col_idx + c] = ""

            # Handle rowspan: track for future rows
            if cell["rowspan"] > 1:
                active_rowspans[(row_idx, col_idx)] = (
                    cell["content"],
                    cell["colspan"],
                    cell["rowspan"] - 1,
                )

            col_idx += cell["colspan"]

        grid.append(grid_row)

    return grid


def _grid_to_markdown(grid: List[List[Optional[str]]]) -> str:
    """
    Convert table grid to markdown format.

    Args:
        grid: 2D list of cell contents

    Returns:
        Markdown formatted table string
    """
    if not grid or not grid[0]:
        return ""

    # Determine column widths
    num_cols = len(grid[0])
    col_widths = [0] * num_cols

    for row in grid:
        for col_idx in range(num_cols):
            if col_idx < len(row):
                cell_content = row[col_idx] or ""
                col_widths[col_idx] = max(col_widths[col_idx], len(cell_content))

    # Ensure minimum width of 3 for markdown separator
    col_widths = [max(width, 3) for width in col_widths]

    # Build markdown rows
    lines = []

    for row_idx, row in enumerate(grid):
        # Build row with proper padding
        cells = []
        for col_idx in range(num_cols):
            if col_idx < len(row):
                content = row[col_idx] or ""
            else:
                content = ""

            # Pad content to column width
            padded = content[: col_widths[col_idx]].ljust(col_widths[col_idx])
            cells.append(padded)

        # Join with pipes
        line = "| " + " | ".join(cells) + " |"
        lines.append(line)

        # Add separator after first row
        if row_idx == 0:
            separator = "|" + "|".join(["-" * (w + 2) for w in col_widths]) + "|"
            lines.append(separator)

    return "\n".join(lines)


# Example usage and testing
if __name__ == "__main__":
    # Test simple table
    simple_html = """
    <table>
        <tr>
            <th>Header 1</th>
            <th>Header 2</th>
        </tr>
        <tr>
            <td>Value 1</td>
            <td>Value 2</td>
        </tr>
    </table>
    """

    print("Simple Table:")
    print(html_to_markdown_table(simple_html))
    print()

    # Test table with colspan
    colspan_html = """
    <table>
        <tr>
            <th colspan="2">Merged Header</th>
        </tr>
        <tr>
            <td>Value 1</td>
            <td>Value 2</td>
        </tr>
    </table>
    """

    print("Table with colspan:")
    print(html_to_markdown_table(colspan_html))
    print()

    # Test multiple tables
    multiple_html = simple_html + colspan_html
    print("Multiple tables:")
    print(html_to_markdown_table(multiple_html))


import json
from pathlib import Path

from input_handlers import load_document
from document_processor import process_part
from summarizer import summarize_document


def main():
    raw_path = input(
        "Enter a PDF, image, DOCX, TXT, or XLSX file path: "
    ).strip().strip('"')

    file_path = Path(raw_path).expanduser()

    file_type, parts = load_document(file_path)

    print(f"\nDetected format: {file_type}")
    print(f"Parts to process: {len(parts)}\n")

    results = []

    # Pass 1: Extract document information.
    for index, part in enumerate(parts, start=1):
        print(f"[{index}/{len(parts)}] Processing {part.label}...")

        try:
            if file_type == "xlsx" and part.structured_data is not None:
                # Excel is already structured. Don't ask AI to recreate it.
                result = {
                    "document_type": "spreadsheet",
                    "text_lines": [],
                    "fields": [],
                    "tables": [part.structured_data],
                    "uncertain_or_unreadable": [],
                }
            else:
                # Use Gemma for other document types as before.
                result = process_part(part)

            results.append({
                "part": part.label,
                "result": result,
            })

            print(
                f"  Extracted "
                f"{len(result.get('fields', []))} fields."
            )

        except Exception as exc:
            print(f"  Extraction failed: {exc}")

            results.append({
                "part": part.label,
                "error": str(exc),
            })

    # Pass 2: Analyze the extracted information.
    analysis = None
    analysis_error = None

    successful_results = [
        item for item in results if "result" in item
    ]

    if successful_results:
        try:
            analysis = summarize_document(
                file_name=file_path.name,
                file_type=file_type,
                part_results=results,
            )

            print("\nAnalysis completed successfully.")

        except Exception as exc:
            analysis_error = str(exc)
            print(f"\nAnalysis failed: {analysis_error}")

    else:
        analysis_error = "No document parts were extracted successfully."

    # Save extraction and analysis together.
    output = {
        "source_file": file_path.name,
        "file_type": file_type,
        "extracted_parts": results,
        "analysis": analysis,
    }

    if analysis_error:
        output["analysis_error"] = analysis_error

    output_path = file_path.with_name(
        f"{file_path.stem}_parsed.json"
    )

    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nProcessing complete.")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()

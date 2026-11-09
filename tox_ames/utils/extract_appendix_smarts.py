from pathlib import Path

import pypdf

pdf_path = Path("benigni_bossa_report.pdf")
reader = pypdf.PdfReader(pdf_path)

# Appendix starts around page 30 (index 29) to page 78 (index 77)
print("Searching for SMARTS or chemical patterns in Appendix...")
found_smarts = []
for idx in range(29, len(reader.pages)):
    text = reader.pages[idx].extract_text()
    if "smarts" in text.lower():
        print(f"Page {idx + 1} mentions SMARTS:")
        lines = text.split("\n")
        for line in lines:
            if "smarts" in line.lower() or "pattern" in line.lower():
                print("   ", line)

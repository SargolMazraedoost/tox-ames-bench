import pypdf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
pdf_path = PROJECT_ROOT / "Thompson et al. - 2025 - AmesFormer State-of-the-Art Mutagenicity Prediction with Graph Transformers-405580.pdf"
reader = pypdf.PdfReader(pdf_path)

print(f"Total pages: {len(reader.pages)}")

# Search for SMARTS case-insensitively across all pages
for idx, page in enumerate(reader.pages, start=1):
    text = page.extract_text()
    if "smarts" in text.lower():
        print(f"=== Page {idx} mentions SMARTS ===")
        lines = text.split("\n")
        for line in lines:
            if "smarts" in line.lower() or "alert" in line.lower():
                print("   ", line)

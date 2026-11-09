import pypdf
from pathlib import Path

pdf_path = Path("benigni_bossa_report.pdf")
reader = pypdf.PdfReader(pdf_path)

# Pages 21 to 28 are index 20 to 27
for idx in range(20, 28):
    text = reader.pages[idx].extract_text()
    print(f"=== Page {idx+1} ===")
    print(text)

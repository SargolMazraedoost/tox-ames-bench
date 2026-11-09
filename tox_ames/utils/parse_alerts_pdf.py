import urllib.request
import pypdf
from pathlib import Path

def download_and_parse():
    pdf_url = "https://publications.jrc.ec.europa.eu/repository/bitstream/JRC43157/eur%20report%20benigni%20130208%20final.pdf"
    pdf_path = Path("benigni_bossa_report.pdf")
    
    print(f"Downloading PDF from {pdf_url}...")
    try:
        urllib.request.urlretrieve(pdf_url, pdf_path)
        print("Download complete.")
    except Exception as e:
        print(f"Download failed: {e}")
        return

    reader = pypdf.PdfReader(pdf_path)
    print(f"PDF Page count: {len(reader.pages)}")
    
    # Let's search for "Structural Alert" or "SA " on all pages and print matching text
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if "structural alert" in text.lower():
            print(f"--- Page {idx} ---")
            lines = text.split("\n")
            for line in lines:
                if any(x in line.lower() for x in ["structural alert", "alert ", "smarts", "sa "]):
                    print(line)

if __name__ == "__main__":
    download_and_parse()

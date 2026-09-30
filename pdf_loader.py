from pypdf import PdfReader
import os

# Reads every .pdf file in folder_path and returns a list of dicts:
#     [{"text": "...", "source": "report.pdf", "page": 1}, ...]
#     One dict per page (empty pages are skipped).

def load_pdfs(folder_path):
    pdfs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder_path, file))
            for page in reader.pages:
                text = page.extract_text()
                if not text or not text.strip():
                    continue
                # pypdf's page_number is 0-based; store the human-readable page.
                pdfs.append({"text": text, "source": file, "page": page.page_number + 1})
    return pdfs

if __name__ == "__main__":
    pdfs = load_pdfs("pdfs")
    print(pdfs)
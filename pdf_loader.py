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
                pdfs.append({"text": page.extract_text(), "source": file, "page": page.page_number})
    return pdfs

if __name__ == "__main__":
    pdfs = load_pdfs("pdfs")
    print(pdfs)
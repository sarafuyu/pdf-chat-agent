from langchain_community.document_loaders import PyPDFLoader

def scan_pdf(pdf_dir="C:/Users/esaydrr/OneDrive - Ericsson/Desktop/dna-projects/pdf-chat-agent/data/pdfs/General_info_etc_2024.pdf"):
    loader = PyPDFLoader(pdf_dir)
    pages = loader.load()

    return pages


pages = scan_pdf()
print(len(pages))
page = pages[0]
print(page.page_content[0:500])
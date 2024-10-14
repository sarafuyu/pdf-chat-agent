from pdf_helpers.helper_pdf_processing import load_pdf

pages = load_pdf('pdf-chat-agent/data/pdfs/General_info_etc_2024.pdf')
print(len(pages))
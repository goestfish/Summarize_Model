import os
from PyPDF2 import PdfReader


def extract_text_from_pdfs(pdf_folder="pdfs", txt_folder="txt"):
    os.makedirs(txt_folder, exist_ok=True)

    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, file_name)
            txt_file_name = os.path.splitext(file_name)[0] + ".txt"
            txt_file_path = os.path.join(txt_folder, txt_file_name)

            try:
                print(f"Processing {file_name}...")
                reader = PdfReader(pdf_path)
                full_text = ""

                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"

                with open(txt_file_path, "w", encoding="utf-8") as txt_file:
                    txt_file.write(full_text)
                print(f"Saved text to {txt_file_path}")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    print(f"Text extraction complete. Extracted texts are saved in '{txt_folder}'.")


extract_text_from_pdfs(pdf_folder="pdfs", txt_folder="txtfiles")
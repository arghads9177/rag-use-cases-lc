import os
import pymupdf  # PyMuPDF

def extract_text_from_pdfs(input_folder: str, output_folder: str):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each PDF in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_folder, filename)
            pdf_name = os.path.splitext(filename)[0]

            try:
                doc = pymupdf.open(pdf_path)
                print(f"Processing: {filename} ({doc.page_count} pages)")

                for page_num,page in enumerate(doc):
                    text = page.get_text()
                    if text.strip():
                        output_filename = f"{pdf_name}_page_{page_num+1}.txt"
                        print(f"Fine name: {output_filename}")
                        output_path = os.path.join(output_folder, output_filename)
                        print(f"Output Path: {output_path}")

                        with open(output_path, "w", encoding="utf-8") as f:
                            f.write(text.strip())
                   

                doc.close()

            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    print("Extraction complete!")

# Example usage
if __name__ == "__main__":
    input_pdf_folder = "pdf-docs"          # Folder containing PDF files
    output_text_folder = "extracted_text_pages"  # Folder to store text files

    extract_text_from_pdfs(input_pdf_folder, output_text_folder)
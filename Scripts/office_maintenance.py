import os
import configparser
from google.generativeai import configure, GenerativeModel
from PIL import Image
from pdf2image import convert_from_path
import fitz  # PyMuPDF
import io

def read_config():
    """Read configuration from config.ini file"""
    config = configparser.ConfigParser()
    if not config.read('config.ini'):
        raise FileNotFoundError("config.ini file not found or empty")
    return config

def ensure_folder_exists(folder_path):
    """Ensure that a folder exists, create if it doesn't"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def extract_bill_data(model, img):
    """
    Use Gemini to extract bill data from an image
    """
    prompt = """
Extract the following information from this bill image in a structured format:
- Date of invoice
- GST number
- Bill number
- Items: list each with Description, QTY, Rate per item
- Total amount

If a field is not present, use N/A.

Output exactly in the following format without additional text:
Date: [date]
GST: [gst]
Bill No: [bill no]
Items:
- Description: [desc], QTY: [qty], Rate: [rate]
- ...
Total: [total]
"""
    try:
        response = model.generate_content([prompt, img])
        print("Gemini Response:")
        print(response.text)
        return response.text
    except Exception as e:
        print(f"Error extracting data with Gemini: {e}")
        return None

def process_pdf_with_gemini(pdf_path, model, output_folder, filename):
    """Process PDF files using Gemini"""
    try:
        print(f"Processing PDF: {os.path.basename(pdf_path)}")
        doc = fitz.open(pdf_path)
        extracted_data = []
        
        for i, page in enumerate(doc):
            print(f"Processing page {i+1} of {os.path.basename(pdf_path)}")
            
            # Render page at high DPI
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            with Image.open(io.BytesIO(img_data)) as img:
                data = extract_bill_data(model, img)
                if data:
                    extracted_data.append(f"Page {i+1}:\n{data}")
        
        doc.close()
        
        if extracted_data:
            output_filename = os.path.splitext(filename)[0] + '.txt'
            output_path = os.path.join(output_folder, output_filename)
            with open(output_path, 'w') as f:
                f.write('\n\n'.join(extracted_data))
            print(f"Saved extracted data: {output_path}")
    except Exception as e:
        print(f"Failed to process PDF {os.path.basename(pdf_path)}: {e}")

def process_image_with_gemini(image_path, model, output_folder, filename):
    """Process image files using Gemini"""
    try:
        img = Image.open(image_path)
        data = extract_bill_data(model, img)
        if data:
            output_filename = os.path.splitext(filename)[0] + '.txt'
            output_path = os.path.join(output_folder, output_filename)
            with open(output_path, 'w') as f:
                f.write(data)
            print(f"Saved extracted data: {output_path}")
    except Exception as e:
        print(f"Failed to process image {filename}: {e}")

def main():
    # Read configuration
    try:
        config = read_config()
        input_folder = config['PATHS']['Input_Folder']
        output_folder = config['PATHS']['Output_Folder']
        api_key = config['API']['Gemini_Key']
    except (KeyError, FileNotFoundError) as e:
        print(f"Configuration error: {e}")
        return
    
    ensure_folder_exists(output_folder)
    
    # Configure Gemini
    configure(api_key=api_key)
    model = GenerativeModel('gemini-1.5-flash')
    
    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        # Skip directories and system files
        if os.path.isdir(file_path) or filename.startswith('.'):
            continue
            
        # Check if file is an image or PDF
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.pdf')
        if not filename.lower().endswith(valid_extensions):
            # Copy non-image files directly to output folder
            import shutil
            shutil.copy2(file_path, os.path.join(output_folder, filename))
            print(f"Copied non-image file: {filename}")
            continue
        
        print(f"Processing: {filename}")
        
        if filename.lower().endswith('.pdf'):
            process_pdf_with_gemini(file_path, model, output_folder, filename)
        else:
            process_image_with_gemini(file_path, model, output_folder, filename)

if __name__ == "__main__":
    main()
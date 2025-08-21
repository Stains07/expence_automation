import os
import configparser
import json
import re
from google.generativeai import configure, GenerativeModel
from PIL import Image
from pdf2image import convert_from_path
import fitz  # PyMuPDF
import io
from datetime import datetime

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
    Use Gemini to extract bill data from an image and return as JSON
    """
    prompt = """
You are an expert invoice data extraction specialist. Your task is to analyze invoice images and extract key information accurately.

1. Identify and extract the following fields:
    - Date of Invoice: Format as DD-MMM-YY or DD/MM/YYYY. If not found, use "N/A".
    - GST Number: Extract the complete GST number. If not found, use "N/A".
    - Bill Number: Extract the complete Bill number or Invoice Number. If not found, use "N/A".
    - Description of Items Purchased: Provide a clear and readable list of items.
    - QTY: Quantity of each item. If not found, use "N/A".
    - Rate per Item: Price of each item. If not found, use "N/A".
    - Total Amount: The final total amount due on the invoice. If not found, use "N/A".
    - Purchaser Name: MFL, Muthoot, or any variation. If any of these names are present, return "Muthoot name: yes". If none are found, return "Muthoot name: no".

2. Output the extracted data in the JSON format below. Ensure it's valid JSON. Do not include any additional text or explanations. If the information is not present, respond with N/A.

{
    "Date of Invoice": "string or N/A",
    "GST Number": "string or N/A",
    "Bill Number": "string or N/A",
    "Items": [
        {
            "Description": "string",
            "QTY": "string or N/A",
            "Rate per Item": "string or N/A"
        }
    ],
    "Total Amount": "string or N/A",
    "Purchaser Name": "Muthoot name: yes or Muthoot name: no"
}
"""
    try:
        response = model.generate_content([prompt, img])
        print("Gemini Raw Response:")
        print(response.text)
        
        # Strip any markdown wrappers like ```json ... ```
        text = response.text.strip()
        if text.startswith('```json'):
            text = text[7:].strip()
        if text.endswith('```'):
            text = text[:-3].strip()
        
        # Parse the response as JSON
        parsed_data = json.loads(text)
        
        # Print parsed JSON to console
        print("Parsed JSON Response:")
        print(json.dumps(parsed_data, indent=2))
        
        return json.dumps(parsed_data, indent=2)
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return json.dumps({
            "Date of Invoice": "N/A",
            "GST Number": "N/A",
            "Bill Number": "N/A",
            "Items": [],
            "Total Amount": "N/A",
            "Purchaser Name": "Muthoot name: no"
        }, indent=2)
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
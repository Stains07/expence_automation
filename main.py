import os
import configparser
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import pytesseract
from pdf2image import convert_from_path
import imutils
import fitz  # PyMuPDF
import io

def read_config():
    """Read configuration from config.ini file"""
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config

def ensure_folder_exists(folder_path):
    """Ensure that a folder exists, create if it doesn't"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def calculate_text_orientation_angle(image):
    """
    Calculate the dominant text orientation angle using OCR
    Returns the angle needed to correct the orientation
    """
    try:
        # Use pytesseract to detect orientation and script
        osd = pytesseract.image_to_osd(image)
        
        # Extract rotation information
        rotation_info = [line for line in osd.split('\n') if 'Rotate: ' in line]
        if rotation_info:
            rotation = int(rotation_info[0].split('Rotate: ')[1])
            return rotation
        return 0
    except Exception as e:
        print(f"Text orientation detection failed: {e}")
        return 0

def needs_rotation_correction(image):
    """
    Determine if the image needs rotation correction
    Only correct if the rotation is significant (more than 5 degrees)
    """
    try:
        # Calculate text orientation
        rotation_angle = calculate_text_orientation_angle(image)
        
        # Only correct if rotation is significant
        if abs(rotation_angle) > 5 and abs(rotation_angle) < 355:
            return True, rotation_angle
        return False, 0
    except Exception as e:
        print(f"Rotation check failed: {e}")
        return False, 0

def correct_image_rotation(image):
    """
    Correct image rotation if text is not in readable orientation
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Check if rotation is needed
        needs_rotation, rotation_angle = needs_rotation_correction(gray)
        
        if needs_rotation:
            print(f"Rotating image by {rotation_angle} degrees")
            # Rotate image to make text horizontal
            rotated = imutils.rotate_bound(image, rotation_angle)
            return rotated
        
        return image
    except Exception as e:
        print(f"Rotation correction failed: {e}")
        return image

def enhance_image_with_previous_filter(image_path):
    """
    Apply the enhancement filter from the previous drinking_water.py code
    This provides the high-contrast black-and-white effect you liked
    """
    try:
        # Read image
        if image_path.lower().endswith('.pdf'):
            # Convert PDF to image
            images = convert_from_path(image_path, dpi=300)
            if images:
                img = images[0].convert("RGB")
            else:
                return None
        else:
            img = Image.open(image_path).convert("RGB")
        
        # Convert to OpenCV format for rotation detection
        img_cv = np.array(img)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        
        # Correct rotation if needed
        img_rotated_cv = correct_image_rotation(img_cv)
        
        # Convert back to PIL
        img_rotated = Image.fromarray(cv2.cvtColor(img_rotated_cv, cv2.COLOR_BGR2RGB))
        
        # Convert to grayscale
        img_gray = img_rotated.convert("L")
        
        # Enhance contrast to make text bright and dark (from drinking_water.py)
        contrast_enhancer = ImageEnhance.Contrast(img_gray)
        img_contrasted = contrast_enhancer.enhance(2.0)  # Increase contrast
        
        # Enhance brightness if needed (from drinking_water.py)
        brightness_enhancer = ImageEnhance.Brightness(img_contrasted)
        img_array = np.array(img_contrasted)
        brightness = brightness_enhancer.enhance(1.5 if np.mean(img_array) < 128 else 1.0)
        
        # Apply black-and-white effect with a lower threshold for high contrast text (from drinking_water.py)
        threshold = 100  # Lower threshold to ensure text is very bright/dark
        img_bw = brightness.point(lambda x: 0 if x < threshold else 255, "L")
        
        # Convert to numpy array for consistency
        img_processed = np.array(img_bw)
        
        return img_processed
    except Exception as e:
        print(f"Error enhancing image with previous filter {image_path}: {e}")
        return None

def process_pdf_with_previous_filter(pdf_path, save_folder):
    """Process PDF files using the previous enhancement filter"""
    try:
        print(f"Processing PDF: {os.path.basename(pdf_path)}")
        doc = fitz.open(pdf_path)
        processed_images = []
        
        for i, page in enumerate(doc):
            print(f"Processing page {i+1} of {os.path.basename(pdf_path)}")
            
            # Render page at high DPI
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            with Image.open(io.BytesIO(img_data)) as img:
                img = img.convert("RGB")
                
                # Convert to OpenCV format for rotation detection
                img_cv = np.array(img)
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
                
                # Correct rotation if needed
                img_rotated_cv = correct_image_rotation(img_cv)
                
                # Convert back to PIL
                img_rotated = Image.fromarray(cv2.cvtColor(img_rotated_cv, cv2.COLOR_BGR2RGB))
                
                # Convert to grayscale
                img_gray = img_rotated.convert("L")
                
                # Enhance contrast to make text bright and dark
                contrast_enhancer = ImageEnhance.Contrast(img_gray)
                img_contrasted = contrast_enhancer.enhance(2.0)
                
                # Enhance brightness if needed
                brightness_enhancer = ImageEnhance.Brightness(img_contrasted)
                img_array = np.array(img_contrasted)
                brightness = brightness_enhancer.enhance(1.5 if np.mean(img_array) < 128 else 1.0)
                
                # Apply black-and-white effect
                threshold = 100
                img_bw = brightness.point(lambda x: 0 if x < threshold else 255, "L")
                
                # Convert to numpy array
                img_processed = np.array(img_bw)
                processed_images.append(img_processed)
        
        doc.close()
        return processed_images
    except Exception as e:
        print(f"Failed to process PDF {os.path.basename(pdf_path)}: {e}")
        return None

def process_office_maintenance_documents():
    """Main function to process Office Maintenance documents using the previous filter"""
    # Read configuration
    config = read_config()
    
    # Get input and output folder paths from config
    try:
        input_folder = config['PATHS']['Office_Maintenance']
        processed_folder = config['PATHS']['Processed_Folder']
    except:
        print("Office_Maintenance or Processed_Folder path not found in config.ini")
        return
    
    # Create processed images folder under the specified Processed_Folder
    output_folder = os.path.join(processed_folder, 'Processed_images')
    ensure_folder_exists(output_folder)
    
    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        # Skip directories and system files
        if os.path.isdir(file_path) or filename.startswith('.'):
            continue
            
        # Check if file is an image or PDF
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.pdf')
        if not filename.lower().endswith(valid_extensions):
            # Copy non-image files directly to processed folder
            import shutil
            shutil.copy2(file_path, os.path.join(output_folder, filename))
            print(f"Copied non-image file: {filename}")
            continue
        
        print(f"Processing: {filename}")
        
        # Process the file based on its type using the previous filter
        if filename.lower().endswith('.pdf'):
            # Process PDF files with previous filter
            processed_images = process_pdf_with_previous_filter(file_path, output_folder)
            
            if processed_images:
                for i, img in enumerate(processed_images):
                    output_filename = f"{os.path.splitext(filename)[0]}_page{i+1}_processed.png"
                    output_path = os.path.join(output_folder, output_filename)
                    cv2.imwrite(output_path, img)
                    print(f"Saved processed PDF page: {output_filename}")
        else:
            # Process image files with previous filter
            enhanced_image = enhance_image_with_previous_filter(file_path)
            
            if enhanced_image is not None:
                # Save the processed image
                output_filename = os.path.splitext(filename)[0] + '_processed.png'
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, enhanced_image)
                print(f"Saved processed image: {output_filename}")
            else:
                print(f"Failed to process: {filename}")

if __name__ == "__main__":
    process_office_maintenance_documents()
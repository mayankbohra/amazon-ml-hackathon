import os
import pandas as pd
import pytesseract
from PIL import Image
from src.utils import download_images
from src.constants import entity_unit_map, allowed_units
import re

TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image(image_path):
    try:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.lower()
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return ""

def parse_entity_value(extracted_text, entity_name):
    pattern = re.compile(r"(\d+(\.\d+)?)\s*([a-zA-Z]+)")
    matches = pattern.findall(extracted_text)
    
    if matches:
        for match in matches:
            value, _, unit = match
            if unit in allowed_units and unit in entity_unit_map.get(entity_name, {}):
                return f"{value} {unit}"
    return ""

def predictor(image_link, category_id, entity_name):
    image_folder = './downloaded_images/'
    image_filename = os.path.basename(image_link)
    image_path = os.path.join(image_folder, image_filename)
    
    if not os.path.exists(image_path):
        download_images([image_link], image_folder, allow_multiprocessing=False)
    
    extracted_text = extract_text_from_image(image_path)
    
    prediction = parse_entity_value(extracted_text, entity_name)
    
    return prediction if prediction else ""

if __name__ == "__main__":
    DATASET_FOLDER = './dataset/'
    IMAGE_DOWNLOAD_FOLDER = './downloaded_images/'
    
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    
    download_images(test['image_link'].tolist(), IMAGE_DOWNLOAD_FOLDER)
    
    test['prediction'] = test.apply(lambda row: predictor(row['image_link'], row['group_id'], row['entity_name']), axis=1)
    
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    test[['index', 'prediction']].to_csv(output_filename, index=False)

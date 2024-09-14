import os
import pandas as pd
from src.utils import download_images
from src.feature_extractor import extract_features
from src.model_trainer import train_model

def check_images_already_downloaded(image_links, image_folder):
    print("Checking if images are already downloaded...")
    missing_images = []
    for image_link in image_links:
        image_filename = os.path.basename(image_link)
        image_path = os.path.join(image_folder, image_filename)
        if not os.path.exists(image_path):
            missing_images.append(image_link)
    print(f"Check completed. {len(missing_images)} missing images found.")
    return missing_images

def predictor(image_link, category_id, entity_name, model, image_folder):
    image_filename = os.path.basename(image_link)
    image_path = os.path.join(image_folder, image_filename)
    
    print(f"Processing image link: {image_link}")
    print(f"Image will be saved at: {image_path}")

    if not os.path.exists(image_path):
        print(f"Image not found locally. Downloading image: {image_filename}")
        download_images([image_link], image_folder, allow_multiprocessing=False)
        print(f"Image downloaded: {image_filename}")
    
    print(f"Extracting features from image: {image_filename}")
    features = extract_features(image_path)

    if features is None:
        print(f"Skipping prediction due to incomplete or corrupt image: {image_filename}")
        return None

    print(f"Making prediction for entity: {entity_name}")
    prediction = model.predict([features])[0]
    print(f"Prediction completed: {prediction}")
    return prediction

if __name__ == "__main__":
    DATASET_FOLDER = './dataset/'
    IMAGE_DOWNLOAD_FOLDER_TRAIN = './downloaded_images/'
    IMAGE_DOWNLOAD_FOLDER_TEST = './downloaded_images_test/'

    print(f"Loading training dataset from folder: {DATASET_FOLDER}")
    train_data = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
    print("Training dataset loaded.")

    missing_train_images = check_images_already_downloaded(train_data['image_link'].tolist(), IMAGE_DOWNLOAD_FOLDER_TRAIN)
    
    if missing_train_images:
        print(f"Found {len(missing_train_images)} images to download for training.")
        download_images(missing_train_images, IMAGE_DOWNLOAD_FOLDER_TRAIN)
        print("Image download process for training completed.")
    else:
        print("All training images are already downloaded.")

    print("Starting feature extraction for training images...")
    X_train = []
    y_train = []
    
    skipped_images = []
    successful_images = []

    for index, row in train_data.iterrows():
        image_link = row['image_link']
        entity_value = row['entity_value'] 
        
        image_filename = os.path.basename(image_link)
        image_path = os.path.join(IMAGE_DOWNLOAD_FOLDER_TRAIN, image_filename)
        
        print(f"Extracting features from image: {image_filename}")
        features = extract_features(image_path)
        
        if features is not None:
            X_train.append(features)
            y_train.append(entity_value)
            successful_images.append(image_filename)
        else:
            skipped_images.append(image_filename)
            print(f"Skipping image: {image_filename} due to incomplete or corrupt file.")

    print("Feature extraction for training completed.")
    print(f"Total images processed: {len(train_data)}")
    print(f"Successfully extracted features from {len(successful_images)} images.")
    print(f"Skipped {len(skipped_images)} images.")
    
    if skipped_images:
        print("Skipped images:")
        for img in skipped_images:
            print(f" - {img}")

    print("Training model on extracted features...")
    model = train_model(X_train, y_train)
    print("Model training completed.")

    print(f"Loading test dataset from folder: {DATASET_FOLDER}")
    test_data = pd.read_csv(os.path.join(DATASET_FOLDER, 'sample_test.csv'))
    print("Test dataset loaded.")

    missing_test_images = check_images_already_downloaded(test_data['image_link'].tolist(), IMAGE_DOWNLOAD_FOLDER_TEST)

    if missing_test_images:
        print(f"Found {len(missing_test_images)} images to download for testing.")
        download_images(missing_test_images, IMAGE_DOWNLOAD_FOLDER_TEST)  
        print("Image download process for testing completed.")
    else:
        print("All test images are already downloaded.")

    print("Starting predictions for test data...")
    test_predictions = test_data.copy()
    test_predictions['prediction'] = test_predictions.apply(
        lambda row: predictor(row['image_link'], row['group_id'], row['entity_name'], model, IMAGE_DOWNLOAD_FOLDER_TEST), axis=1
    )

    test_predictions = test_predictions[test_predictions['prediction'].notnull()]

    print("Predictions for test data completed.")

    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    print(f"Saving predictions to {output_filename}...")
    test_predictions[['index', 'prediction']].to_csv(output_filename, index=False)
    print(f"Predictions saved to {output_filename}.")

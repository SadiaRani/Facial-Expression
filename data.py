import cv2
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)  # Load images as grayscale
        if img is not None:
            images.append(img)
    return images

def normalize_contrast(images):
    normalized_images = [cv2.equalizeHist(img) for img in images]
    return normalized_images

def normalize_images_in_folders(root_folder):
    for dataset_folder in os.listdir(root_folder):
        dataset_path = os.path.join(root_folder, dataset_folder)
        if os.path.isdir(dataset_path):
            print("Normalizing images in", dataset_folder, "folder...")
            for emotion_folder in os.listdir(dataset_path):
                emotion_path = os.path.join(dataset_path, emotion_folder)
                if os.path.isdir(emotion_path):
                    images = load_images_from_folder(emotion_path)
                    normalized_images = normalize_contrast(images)
                    # Save normalized images back to the same folder
                    for i, img in enumerate(normalized_images):
                        cv2.imwrite(os.path.join(emotion_path, f"normalized_{i}.jpg"), img)
            print("Normalization completed for", dataset_folder, "folder.\n")

# Path to the root folder containing emotion categories (Facial expression)
root_folder_path = r'C:\Users\lenovo\PycharmProjects\Opencvproject\Facial Recognition Dataset'

# Normalize images in each dataset folder (training, testing, validation)
normalize_images_in_folders(root_folder_path)


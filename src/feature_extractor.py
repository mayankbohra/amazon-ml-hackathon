import torch
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
import os

resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

def preprocess_for_resnet(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0)  
        return img_tensor
    except (OSError, UnidentifiedImageError) as e:
        return None

def extract_features(image_path):
    img_tensor = preprocess_for_resnet(image_path)
    if img_tensor is None:
        return None  
    with torch.no_grad():
        features = resnet(img_tensor)
    return features.squeeze().numpy()  

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from models_definition.efficient_net_new import EfficientnetCENew
from models_definition.efficient_arcface import efficient_net_arcface
from models_definition.triplet_efficientnet_model import TripletModel
from facenet_pytorch import InceptionResnetV1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((160, 160)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

### helper function
def encode(model_name, model, img):
    if model_name == 'ce-efficientnet':
        img = transform(img[0])
        res = model(img.unsqueeze(0), return_embeddings=True)
    elif model_name == 'arcface-efficientnet':
        img = transform(img[0])
        res = model(img.unsqueeze(0))
        res = F.normalize(res, p=2, dim=1)
    elif model_name in ['ce-inceptionresnetv1-1292', 'ce-inceptionresnetv1-822',
                        'triplet-efficientnet-min-val-loss', 'triplet-efficientnet-last-epoch']:
        img = transform(img[0])
        res = model(img.unsqueeze(0))

    return res

def load_ce_efficientnet(path, num_identities):
    model = EfficientnetCENew(512, num_identities)
    checkpoint = torch.load(path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def load_arface_efficientnet(path, num_identities):
    model = efficient_net_arcface
    checkpoint = torch.load(path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def load_ce_inceptionresnetv1(path, num_identities):
    model = InceptionResnetV1(pretrained=None, num_classes=num_identities, classify=True, device=device)
    model.logits = nn.Linear(in_features=512, out_features=num_identities)
    checkpoint = torch.load(path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.classify = False
    return model

def load_triplet_efficientnet(path, num_identities):
    triplet_model = TripletModel(512, 1292)
    checkpoint = torch.load(path, weights_only=False, map_location=device)
    triplet_model.load_state_dict(checkpoint['model_state_dict'])
    return triplet_model

# Function to compute area of the bounding box
def compute_area(box):
    if box is None:
        return 0
    x_min, y_min, x_max, y_max = box
    return (x_max - x_min) * (y_max - y_min)

def clamp_box(box, img_width, img_height):
    x_min, y_min, x_max, y_max = box

    # Clamp coordinates to be within the image dimensions
    x_min = max(0, min(x_min, img_width - 1))
    y_min = max(0, min(y_min, img_height - 1))
    x_max = max(0, min(x_max, img_width - 1))
    y_max = max(0, min(y_max, img_height - 1))

    return [x_min, y_min, x_max, y_max]

# Function to align the image based on eye landmarks
def align_face(image, landmarks):
    # Get the left and right eye coordinates
    left_eye = landmarks[0]
    right_eye = landmarks[1]

    # Compute the angle between the eyes
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # Compute the center of both eyes (for rotation)
    eye_center = ((left_eye[0] + right_eye[0]) // 2,
                  (left_eye[1] + right_eye[1]) // 2)

    # Get the rotation matrix
    rot_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)

    # Rotate the entire image
    aligned_image = cv2.warpAffine(image, rot_matrix, (image.shape[0], image.shape[1]))

    # Rotate the bounding box coordinates (if needed later)
    return aligned_image

# Function to apply rotation matrix to the bounding box
def rotate_box(box, rot_matrix):
    # Extract the four corners of the bounding box
    x_min, y_min, x_max, y_max = box
    corners = np.array([[x_min, y_min], [x_min, y_max], [x_max, y_min], [x_max, y_max]])

    # Apply the rotation matrix to the four corners
    ones = np.ones(shape=(len(corners), 1))
    corners_ones = np.hstack([corners, ones])

    # Apply the rotation
    transformed_corners = rot_matrix.dot(corners_ones.T).T

    # Get the new bounding box coordinates
    x_min_new, y_min_new = np.min(transformed_corners, axis=0)[:2]
    x_max_new, y_max_new = np.max(transformed_corners, axis=0)[:2]

    return [x_min_new, y_min_new, x_max_new, y_max_new]
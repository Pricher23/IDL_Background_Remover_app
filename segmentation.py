import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

mp_selfie = mp.solutions.selfie_segmentation

_custom_model = None
# ImageNet mean/std normalization, same values used during training
_img_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_custom_model(path="model_weights.pth"):
    global _custom_model
    backbone = models.mobilenet_v3_small(weights=None)
    # replace the classifier head to output a flat 256x256 mask
    backbone.classifier = nn.Sequential(
        nn.Linear(576, 256),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(256, 256 * 256),
        nn.Sigmoid(),
    )
    backbone.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    backbone.eval()
    _custom_model = backbone
    print("Custom model loaded.")

def apply_blur_background(frame, mask, blur_strength=51):
    blurred = cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)
    # need 3 channels to blend with the color frame
    mask_3ch = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
    result = (frame.astype(np.float32) * mask_3ch +
              blurred.astype(np.float32) * (1.0 - mask_3ch))
    return result.astype(np.uint8)

def get_mask_custom(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = _img_transform(img).unsqueeze(0)
    with torch.no_grad():
        mask = _custom_model(tensor).view(256, 256).numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    # blur before thresholding makes edges softer, found this helps a lot
    mask = cv2.GaussianBlur(mask, (21, 21), 0)

    # tried 10x10 rect kernel first but hair had holes, ellipse at 15 worked better
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # close holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # TODO: maybe try dilating slightly instead of just closing?
    return mask

def get_mask(frame, model):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = model.process(rgb)
    mask = result.segmentation_mask
    # 0.6 threshold works better than 0.5 for mediapipe, less noise
    mask = (mask > 0.6).astype(np.uint8) * 255
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    return mask

def apply_background(frame, mask, background):
    bg = cv2.resize(background, (frame.shape[1], frame.shape[0]))
    mask_3ch = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
    result = (frame.astype(np.float32) * mask_3ch +
              bg.astype(np.float32) * (1.0 - mask_3ch))
    return result.astype(np.uint8)

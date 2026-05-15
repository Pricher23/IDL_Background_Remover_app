import cv2
import os
import numpy as np

def load_backgrounds(folder="backgrounds"):
    images = {}
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(f)[0]
            img = cv2.imread(os.path.join(folder, f))
            if img is not None:
                images[name] = img
    return images

def solid_color_background(h, w, color=(0, 200, 0)):
    import numpy as np
    bg = np.zeros((h, w, 3), dtype="uint8")
    bg[:] = color
    return bg

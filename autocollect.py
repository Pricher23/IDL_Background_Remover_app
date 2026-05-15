import cv2
import mediapipe as mp
import numpy as np
import os
import time

mp_selfie = mp.solutions.selfie_segmentation

os.makedirs("dataset/frames", exist_ok=True)
os.makedirs("dataset/masks", exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0
target = 150
prev_time = time.time()

print(f"Auto-capturing {target} frames, one per second.")

with mp_selfie.SelfieSegmentation(model_selection=1) as model:
    while cap.isOpened() and count < target:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = model.process(rgb)
        mask = (result.segmentation_mask > 0.6).astype(np.uint8) * 255

        now = time.time()
        # dont capture too fast
        if now - prev_time >= 0.5:
            frame_resized = cv2.resize(frame, (256, 256))
            mask_resized = cv2.resize(mask, (256, 256))
            cv2.imwrite(f"dataset/frames/{count:04d}.jpg", frame_resized)
            cv2.imwrite(f"dataset/masks/{count:04d}.png", mask_resized)
            count += 1
            prev_time = now
            print(f"Captured {count}/{target}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"Done! Collected {count} frames in dataset/")

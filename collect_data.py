import cv2
import mediapipe as mp
import numpy as np
import os

mp_selfie = mp.solutions.selfie_segmentation

os.makedirs("dataset/frames", exist_ok=True)
os.makedirs("dataset/masks", exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("Collecting frames... Press SPACE to capture, Q to quit.")

with mp_selfie.SelfieSegmentation(model_selection=1) as model:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = model.process(rgb)
        mask = (result.segmentation_mask > 0.6).astype(np.uint8) * 255

        # show green overlay so you can see what mediapipe is picking up
        preview = frame.copy()
        preview[mask == 0] = [0, 200, 0]
        cv2.putText(preview, f"Captured: {count} | SPACE=capture Q=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow("Collecting Data", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            # save both at 256x256 to match training input size
            frame_resized = cv2.resize(frame, (256, 256))
            mask_resized = cv2.resize(mask, (256, 256))
            cv2.imwrite(f"dataset/frames/{count:04d}.jpg", frame_resized)
            cv2.imwrite(f"dataset/masks/{count:04d}.png", mask_resized)
            count += 1
            print(f"Captured frame {count}")
        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"Done! Collected {count} frames.")

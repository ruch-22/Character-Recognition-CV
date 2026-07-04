from function import *
import cv2
import os
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands

os.makedirs(DATA_PATH, exist_ok=True)

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
) as hands:

    for action in actions:

        os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)

        for image_num in range(no_sequences):

            image_path = f"Data/{action}/{image_num}.png"

            frame = cv2.imread(image_path)

            if frame is None:
                print(f"Could not read {image_path}")
                continue

            image, results = mediapipe_detection(frame, hands)

            draw_styles_landmarks(image, results)

            keypoints = extract_keypoints(results)

            np.save(
                os.path.join(DATA_PATH, action, f"{image_num}.npy"),
                keypoints
            )

            cv2.imshow("Image", image)

            cv2.waitKey(50)

cv2.destroyAllWindows()

print("Dataset created successfully!")
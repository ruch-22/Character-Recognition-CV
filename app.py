from function import *

import cv2
import numpy as np

from tensorflow.keras.models import load_model

# Load trained model
model = load_model("model.keras")

# Colors for prediction bars
colors = [
    (245, 117, 16),
    (117, 245, 16),
    (16, 117, 245)
]

threshold = 0.80

# Visualization
def prob_viz(res, actions, image, colors):

    output = image.copy()

    for num, prob in enumerate(res):

        cv2.rectangle(
            output,
            (0, 60 + num * 40),
            (int(prob * 300), 90 + num * 40),
            colors[num],
            -1
        )

        cv2.putText(
            output,
            f"{actions[num]} : {prob:.2f}",
            (10, 85 + num * 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    return output


cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # ROI
        cropped = frame[40:400, 0:300]

        cv2.rectangle(
            frame,
            (0, 40),
            (300, 400),
            (255, 0, 0),
            2
        )

        image, results = mediapipe_detection(cropped, hands)

        draw_styles_landmarks(image, results)

        keypoints = extract_keypoints(results)

        # Predict
        res = model.predict(
            np.expand_dims(keypoints, axis=0),
            verbose=0
        )[0]

        predicted_class = np.argmax(res)

        confidence = res[predicted_class]

        if confidence > threshold:

            cv2.putText(
                frame,
                f"{actions[predicted_class]} ({confidence*100:.1f}%)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

        frame = prob_viz(res, actions, frame, colors)

        cv2.imshow("Hand Sign Recognition", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
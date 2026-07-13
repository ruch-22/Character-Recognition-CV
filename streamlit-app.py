import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model

from function import *

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Hand Sign Recognition",
    page_icon="🖐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Load Model
# -------------------------------------------------

@st.cache_resource
def load_sign_model():
    try:
        return load_model("model.keras", compile=False)
    except Exception as e:
        st.error(f"Unable to load model: {e}")
        st.stop()

model = load_sign_model()

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.title("🖐 Hand Sign Recognition")

st.sidebar.success("Model Loaded Successfully")

st.sidebar.markdown("---")

st.sidebar.subheader("Supported Signs")

for sign in actions:
    st.sidebar.write(f"✅ {sign}")

st.sidebar.markdown("---")

st.sidebar.subheader("Technologies Used")

st.sidebar.write("• TensorFlow")
st.sidebar.write("• MediaPipe")
st.sidebar.write("• OpenCV")
st.sidebar.write("• Streamlit")

st.sidebar.markdown("---")

st.sidebar.info(
    """
Current Version: **1.0**

Supports recognition of:

- A
- B
- C
"""
)

# -------------------------------------------------
# Main Title
# -------------------------------------------------
st.title("🖐 Hand Sign Recognition")

st.write(
    """
Detect static hand signs using **MediaPipe** and a **TensorFlow Deep Neural Network**.
"""
)

st.markdown("---")

# -------------------------------------------------
# Upload Image
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload a Hand Sign Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:

        processed_image, results = mediapipe_detection(image_bgr, hands)

        draw_styles_landmarks(processed_image, results)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")

            st.image(
                image,
                use_container_width=True
            )

        with col2:
            st.subheader("Detected Landmarks")

            st.image(
                cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

        st.markdown("---")

        if results.multi_hand_landmarks:

            keypoints = extract_keypoints(results)

            input_data = np.expand_dims(keypoints, axis=0)

            prediction = model.predict(
                input_data,
                verbose=0
            )[0]

            predicted_index = np.argmax(prediction)

            predicted_letter = actions[predicted_index]
            
            if model.output_shape[-1] != len(actions):
                st.error(
                    f"Model expects {model.output_shape[-1]} classes, "
                    f"but actions has {len(actions)}."
                )
                st.stop()

            confidence = prediction[predicted_index] * 100

            st.success("✅ Hand Detected")

            metric1, metric2 = st.columns(2)

            with metric1:
                st.metric(
                    "Detected Sign",
                    predicted_letter
                )

            with metric2:
                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            st.markdown("---")

            st.subheader("Prediction Probabilities")

            probability_df = pd.DataFrame({
                "Class": actions,
                "Probability": prediction
            }).sort_values(
                by="Probability",
                ascending=False
            )

            st.bar_chart(
                probability_df.set_index("Class")
            )

            st.markdown("### Detailed Confidence")

            for i, sign in enumerate(actions):

                st.write(f"**{sign}**")

                st.progress(float(prediction[i]))

                st.write(f"{prediction[i]*100:.2f}%")

        else:

            st.error("❌ No hand detected.")

            st.info(
                """
Please upload a clearer image where:

- One hand is visible
- The entire hand is inside the image
- Lighting is sufficient
"""
            )
            
confidence = prediction[predicted_index]

if confidence < 0.60:
    st.warning(
        "Low confidence prediction. "
        "Please upload a clearer image."
    )
else:
    predicted_letter = actions[predicted_index]

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")

st.subheader("Project Information")

col1, col2 = st.columns(2)

with col1:
    st.write("**Model**")
    st.write("Deep Neural Network")

    st.write("**Framework**")
    st.write("TensorFlow + MediaPipe")

with col2:
    st.write("**Input Features**")
    st.write("63 Hand Landmark Features")

    st.write("**Supported Classes**")
    st.write(", ".join(actions))

st.markdown("---")

st.caption(
    "Developed by **Ruchika Bambal** | AI • Computer Vision • TensorFlow"
)
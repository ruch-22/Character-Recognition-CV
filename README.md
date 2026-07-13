# 🖐️ Hand Sign Recognition using MediaPipe and TensorFlow

A real-time hand sign recognition system that detects static hand gestures using **MediaPipe Hands** and classifies them using a **TensorFlow Deep Neural Network (DNN)**. The project recognizes hand signs such as **A, B, and C** from a webcam feed.

---

## 📌 Features

- Real-time hand detection using MediaPipe
- Hand landmark extraction (21 landmarks, 63 features)
- Deep Neural Network for hand sign classification
- Live webcam prediction
- Confidence score visualization
- Easy to extend for additional hand signs

---

## 📂 Project Structure

```
WasteDetection/
│
├── Data/
│   ├── A/
│   ├── B/
│   └── C/
│
├── MP_Data/
│   ├── A/
│   ├── B/
│   └── C/
│
├── Logs/
│
├── function.py
├── datacollection.py
├── data.py
├── trainmodel.py
├── app.py
├── model.keras
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies Used

- Python 3.10+
- TensorFlow / Keras
- MediaPipe
- OpenCV
- NumPy
- Scikit-learn

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Hand-Sign-Recognition.git

cd Hand-Sign-Recognition
```

---

### 2. Create a virtual environment

Windows

```bash
python -m venv .venv
```

Activate

```bash
.venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

The dataset consists of static hand images.

Classes:

- A
- B
- C

Each class contains approximately **30 images**.

The images are captured using the webcam and stored in the `Data` folder.

---

## 🚀 Project Workflow

### Step 1: Collect Images

Run

```bash
python datacollection.py
```

This captures hand images for each gesture.

---

### Step 2: Extract Hand Keypoints

Run

```bash
python data.py
```

This extracts

- 21 hand landmarks
- x, y, z coordinates

Total Features:

```
21 × 3 = 63
```

The extracted features are stored as `.npy` files inside `MP_Data`.

---

### Step 3: Train the Model

Run

```bash
python trainmodel.py
```

This

- Loads all keypoints
- Splits the dataset
- Trains a Deep Neural Network
- Saves the trained model as

```
model.keras
```

---

### Step 4: Run Real-Time Recognition

Run

```bash
python app.py
```

The webcam opens and predicts the detected hand sign in real time.

Press

```
Q
```

to quit.

---

## 🧠 Model Architecture

```
Input Layer
      │
      ▼
Dense (128)
      │
Dropout (0.3)
      │
Dense (64)
      │
Dropout (0.3)
      │
Dense (32)
      │
Output Layer
(Softmax)
```

Input Features:

```
63
```

Output Classes:

```
3
```

---

## 📷 Hand Landmark Detection

MediaPipe detects **21 hand landmarks**.

Each landmark contains

- x
- y
- z

Total features:

```
63
```

These features are used for training instead of raw images.

---

## 📈 Future Improvements

- Add all 26 alphabet signs
- Add digit recognition (0–9)
- Improve dataset size
- Support two-hand gestures
- Build sentence recognition
- Add speech output
- Deploy as a web application using Streamlit
- Deploy using Flask or FastAPI

---

## 📦 Requirements

Example

```
tensorflow==2.15.0
mediapipe==0.10.14
opencv-python
numpy
scikit-learn
matplotlib
```

---

## 👩‍💻 Author

**Ruchika Bambal**

B.Tech Student

Interested in AI, Machine Learning, Computer Vision, and Web Development.

---

## ⭐ Acknowledgements

- Google MediaPipe
- TensorFlow
- OpenCV
- Scikit-learn

---

## 📜 License

This project is created for educational and learning purposes.

Feel free to use and modify it.

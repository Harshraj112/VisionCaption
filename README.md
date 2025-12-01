# 🖼️ Image Captioning Project (CNN + LSTM)

This project implements an **Image Caption Generator** using:
- Xception CNN for image feature extraction
- LSTM network for sequence modeling
- Flickr8k dataset for training captions

The model generates textual descriptions of images based on visual features.

---

## 📂 Project Structure

├── Flicker8k_Dataset/ # Image dataset (NOT in repo) 
├── Flickr8k_text/ # Caption files (NOT in repo) 
├── models2/ # Trained model weights (NOT in repo) 
├── static/ # For website use (Flask/Streamlit)
├── test.py # Run inference
├── model.py # Training script
├── tokenizer.p # Saved tokenizer
├── descriptions.txt # Clean captions
├── features.p # Image features
├── requirements.txt
├── .gitignore
└── README.md


---

## 🚀 How to Setup

### 1️⃣ Create virtual environment (Python 3.10 or 3.11)
python3.11 -m venv tfenv
source tfenv/bin/activate


### 2️⃣ Install dependencies
pip install -r requirements.txt


---

## 🏋️ Model Training

Run:
python model.py


This will:
- Load Flickr8k captions
- Preprocess text
- Extract Xception features
- Train CNN+LSTM model
- Save weights in `models2/`

---

## 🧪 Test the Model

Use your image:
python test.py --image YOUR_IMAGE.jpg


Example:
python test.py --image Flicker8k_Dataset/1000268201_693b08cb0e.jpg


---

## ✅ Example Output
man in black shirt is standing in front of woman


---

## ⚠️ Important Notes

✅ Works well on Flickr8K images  
⚠️ Performance may drop for real-world images due to limited training data  
⚠️ This is an academic model, not production-ready

---

## 🧠 Model Architecture

- Xception (CNN) for feature extraction
- LSTM for caption decoding
- Softmax vocabulary prediction
- Trained with teacher forcing

Model diagram is stored in `model.png`.

---

## 🔮 Future Improvements

- Train on MS COCO or Flickr30k
- Use attention mechanism
- Add Beam Search
- Use transformer-based captioning
- Create web interface

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Xception CNN
- LSTM
- NumPy
- Matplotlib

---

## 👨‍💻 Author

**Harsh Raj**  
Image Captioning Project for learning Deep Learning & Computer Vision

---

## ⭐ If you like this project, give it a star!

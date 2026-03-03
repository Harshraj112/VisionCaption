import streamlit as st
import numpy as np
from PIL import Image
from pickle import load
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.models import Model
import os
import urllib.request
import ssl
import gdown

# ─── Page Config ───
st.set_page_config(
    page_title="VisionCaption – Image Caption Generator",
    page_icon="🖼️",
    layout="centered",
)

# ─── Constants ───
MAX_LENGTH = 32
MODEL_DIR = "models2"
MODEL_PATH = os.path.join(MODEL_DIR, "model_9.h5")
TOKENIZER_PATH = "tokenizer.p"

# Google Drive file ID for model weights (update this with your file ID)
GDRIVE_MODEL_ID = "1GYeMUq2u2kBRsNzOyNquBP1mgMbCQgfw"


# ─── Model Architecture ───
@st.cache_resource(show_spinner="Loading caption model...")
def build_caption_model(vocab_size, max_length):
    """Recreate the CNN+LSTM caption model architecture."""
    inputs1 = Input(shape=(2048,), name="input_1")
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation="relu")(fe1)

    inputs2 = Input(shape=(max_length,), name="input_2")
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation="relu")(decoder1)
    outputs = Dense(vocab_size, activation="softmax")(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


@st.cache_resource(show_spinner="Loading Xception feature extractor...")
def load_xception():
    """Load pre-trained Xception (without top) for feature extraction."""
    return Xception(include_top=False, pooling="avg")


@st.cache_resource(show_spinner="Loading tokenizer...")
def load_tokenizer():
    """Load the saved tokenizer."""
    return load(open(TOKENIZER_PATH, "rb"))


def download_model_weights():
    """Download model weights from Google Drive if not present locally."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights from Google Drive..."):
            url = f"https://drive.google.com/uc?id={GDRIVE_MODEL_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)


# ─── Feature Extraction ───
def extract_features(image: Image.Image, model):
    """Extract 2048-d feature vector from an image using Xception."""
    image = image.resize((299, 299))
    img_array = np.array(image)
    # Handle RGBA images
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 127.5 - 1.0
    return model.predict(img_array, verbose=0)


# ─── Caption Generation ───
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_caption(model, tokenizer, photo, max_length):
    """Generate a caption for the given photo features."""
    in_text = "start"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == "end":
            break
    # Remove start/end tokens
    caption = in_text.replace("start", "").replace("end", "").strip()
    return caption


# ─── UI ───
st.title("🖼️ VisionCaption")
st.markdown("**AI-powered Image Caption Generator** using CNN (Xception) + LSTM")
st.divider()

# Download model weights if missing
download_model_weights()

# Load resources
tokenizer = load_tokenizer()
vocab_size = len(tokenizer.word_index) + 1
xception_model = load_xception()
caption_model = build_caption_model(vocab_size, MAX_LENGTH)

# Load weights
if os.path.exists(MODEL_PATH):
    caption_model.load_weights(MODEL_PATH)
else:
    st.error(
        f"Model weights not found at `{MODEL_PATH}`. "
        "Please place `model_9.h5` inside the `models2/` directory."
    )
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"], help="Supported: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Generating caption..."):
        photo_features = extract_features(image, xception_model)
        caption = generate_caption(caption_model, tokenizer, photo_features, MAX_LENGTH)

    st.success("Caption generated!")
    st.markdown(f"### 📝 *{caption.capitalize()}*")
else:
    st.info("👆 Upload an image to generate a caption.")

# ─── Sidebar ───
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        **VisionCaption** generates natural language descriptions 
        of images using a deep learning model trained on the 
        **Flickr8k** dataset.

        **Architecture:**
        - **Xception** CNN → 2048-d image features
        - **LSTM** → sequence generation
        - Trained for 10 epochs

        [GitHub Repository](https://github.com/Harshraj112/VisionCaption)
        """
    )

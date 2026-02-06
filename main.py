from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import streamlit as st
from PIL import Image

# Keras 3+ saving/loading API (recommended)
import keras


# ---- Configuration ----

@dataclass(frozen=True)
class AppConfig:
    model_filename: str = "resnet50_cifar100.keras.zip"
    image_size: tuple[int, int] = (128, 128)
    top_k: int = 5


CIFAR100_LABELS: list[str] = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle",
    "mountain", "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon",
    "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail",
    "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone",
    "television", "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree",
    "wolf", "woman", "worm",
]


def candidate_model_paths(model_filename: str) -> list[Path]:
    """
    Streamlit can run with a different working directory than your code.
    These candidates cover common layouts:
      - model in project root
      - model next to this file
      - model in /models
      - app in /pages and model in project root
    """
    here = Path(__file__).resolve()
    return [
        Path.cwd() / model_filename,
        here.parent / model_filename,
        here.parent.parent / model_filename,
        Path.cwd() / "models" / model_filename,
        here.parent / "models" / model_filename,
        here.parent.parent / "models" / model_filename,
    ]


def first_existing_path(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


@st.cache_resource(show_spinner="Loading model...")
def load_model(model_path: str):
    # Use the Keras 3 load API; compile=False for inference-only apps.
    # safe_mode=True is default and recommended.
    return keras.saving.load_model(model_path, compile=False, safe_mode=True)


def preprocess_image(img: Image.Image, image_size: tuple[int, int]) -> np.ndarray:
    # NOTE: This normalizes to [0,1]. If your training did NOT normalize,
    # remove the division to match training exactly.
    img = img.convert("RGB").resize(image_size)
    x = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(x, axis=0)  # (1, H, W, 3)


def predict_topk(model, x: np.ndarray, labels: list[str], k: int) -> list[tuple[str, float]]:
    probs = model.predict(x, verbose=0)[0]  # shape (num_classes,)
    idxs = np.argsort(probs)[::-1][:k]
    return [(labels[int(i)], float(probs[int(i)])) for i in idxs]


def main():
    st.set_page_config(page_title="CIFAR-100 Classifier", page_icon="🧠", layout="centered")
    cfg = AppConfig()

    st.title("CIFAR-100 Image Classifier")
    st.caption(f"Model file: `{cfg.model_filename}` (expected in project root or a `models/` folder)")

    model_path = first_existing_path(candidate_model_paths(cfg.model_filename))
    if model_path is None:
        st.error("Model file not found.")
        st.code("\n".join(str(p) for p in candidate_model_paths(cfg.model_filename)))
        st.stop()

    try:
        model = load_model(str(model_path))
    except Exception as e:
        st.error("Failed to load the model. Full error:")
        st.exception(e)
        st.stop()

    uploaded = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        st.info("Upload an image to get a prediction.")
        st.stop()

    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_container_width=True)

    x = preprocess_image(img, cfg.image_size)
    topk = predict_topk(model, x, CIFAR100_LABELS, cfg.top_k)

    top1_label, top1_prob = topk[0]
    st.success(f"Predicted: {top1_label} ({top1_prob * 100:.1f}%)")
    st.caption(f"Loaded from: `{model_path}`")

    st.subheader(f"Top {cfg.top_k}")
    for label, prob in topk:
        st.write(f"{label} — {prob * 100:.1f}%")


if __name__ == "__main__":
    main()

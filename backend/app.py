from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf

from utils.preprocess import preprocess_landmarks

app = Flask(__name__)
CORS(app)

# =========================
# Load models
# =========================
arabic_letters_model = tf.keras.models.load_model(
    "model/old_arabic_model.keras",
    compile=False
)

english_letters_model = tf.keras.models.load_model(
    "model/english_all.keras",
    compile=False
)

arabic_words_model = tf.keras.models.load_model(
    "model/sign_model.keras",
    compile=False
)

# =========================
# Helpers for model shapes
# =========================
def get_model_input_shape(model):
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    return shape


ARABIC_LETTERS_INPUT_SHAPE = get_model_input_shape(arabic_letters_model)
ENGLISH_INPUT_SHAPE = get_model_input_shape(english_letters_model)
ARABIC_WORDS_INPUT_SHAPE = get_model_input_shape(arabic_words_model)

# Expected:
# arabic letters / english: (None, 63)
# arabic words LSTM: (None, timesteps, 63)
ARABIC_WORDS_TIMESTEPS = (
    ARABIC_WORDS_INPUT_SHAPE[1]
    if len(ARABIC_WORDS_INPUT_SHAPE) == 3 and ARABIC_WORDS_INPUT_SHAPE[1] is not None
    else 30
)
ARABIC_WORDS_FEATURES = (
    ARABIC_WORDS_INPUT_SHAPE[2]
    if len(ARABIC_WORDS_INPUT_SHAPE) == 3
    else 63
)

# =========================
# Load encoders
# =========================
def load_encoder_labels(path, key_candidates):
    loaded = np.load(path, allow_pickle=True)

    # plain numpy array
    if isinstance(loaded, np.ndarray) and loaded.dtype != object:
        return loaded

    # object dict
    try:
        loaded = loaded.item()
    except Exception:
        return loaded

    for key in key_candidates:
        if key in loaded:
            value = loaded[key]
            if isinstance(value, dict):
                return {int(k): v for k, v in value.items()}
            return value

    raise ValueError(f"Unsupported encoder format for {path}")


arabic_labels = load_encoder_labels(
    "model/label_encoder_old.npy",
    ["letters", "idx_to_letter", "idx_to_char"]
)

english_labels = load_encoder_labels(
    "model/english_all_encoder.npy",
    ["letters", "idx_to_letter", "idx_to_char"]
)

arabic_words_labels = load_encoder_labels(
    "model/arabic_words_encoder.npy",
    ["words", "idx_to_word", "labels"]
)

# =========================
# Arabic transliteration map
# =========================
ARABIC_MAP = {
    "aleff": "ا",
    "bb": "ب",
    "taa": "ت",
    "thaa": "ث",
    "jeem": "ج",
    "haa": "ح",
    "khaa": "خ",
    "dal": "د",
    "thal": "ذ",
    "ra": "ر",
    "zay": "ز",
    "seen": "س",
    "sheen": "ش",
    "saad": "ص",
    "dhad": "ض",
    "ta": "ط",
    "dha": "ظ",
    "ain": "ع",
    "ghain": "غ",
    "fa": "ف",
    "gaaf": "ق",
    "kaaf": "ك",
    "laam": "ل",
    "meem": "م",
    "nun": "ن",
    "ha": "ه",
    "waw": "و",
    "ya": "ي",
    "al": "ال",
    "la": "لا",
    "toot": "ة",
    "yaa": "ى",

}
IDX_TO_ARABIC_WORDS = {
    0: "انا",
    1: "اسمي",
    2: "مرحبا",
    3: "عمري",
    4: "سنة",
}
# =========================
# Input normalization
# =========================
def normalize_input_landmarks(landmarks):
    if isinstance(landmarks, float):
        raise ValueError("Landmarks must not be a float value.")

    if not isinstance(landmarks, list) or len(landmarks) == 0:
        raise ValueError("Landmarks must be a non-empty list.")

    first_item = landmarks[0]

    # flat list [x1,y1,z1,...]
    if isinstance(first_item, (int, float)):
        if len(landmarks) != 63:
            raise ValueError("Flat landmarks array must contain exactly 63 values.")
        return [float(v) for v in landmarks]

    # object list [{"x":..,"y":..,"z":..}, ...]
    if isinstance(first_item, dict):
        if len(landmarks) != 21:
            raise ValueError("Object landmarks list must contain exactly 21 points.")

        flat_landmarks = []
        for point in landmarks:
            if not all(k in point for k in ("x", "y", "z")):
                raise ValueError("Each landmark object must contain x, y, and z.")

            flat_landmarks.extend([
                float(point["x"]),
                float(point["y"]),
                float(point["z"]),
            ])
        return flat_landmarks

    raise ValueError("Unsupported landmarks format.")


def preprocess_single_frame(landmarks):
    normalized = normalize_input_landmarks(landmarks)
    processed = preprocess_landmarks(normalized)
    processed = np.array(processed, dtype=np.float32)

    if processed.ndim != 1:
        processed = processed.reshape(-1)

    if processed.shape[0] != 63:
        raise ValueError(
            f"Processed frame must contain exactly 63 values, got shape {processed.shape}"
        )

    return processed


def preprocess_sequence_frames(sequence_landmarks):
    if not isinstance(sequence_landmarks, list) or len(sequence_landmarks) == 0:
        raise ValueError("sequence_landmarks must be a non-empty list of frames.")

    processed_frames = []
    for frame in sequence_landmarks:
        processed_frame = preprocess_single_frame(frame)
        processed_frames.append(processed_frame)

    sequence = np.array(processed_frames, dtype=np.float32)

    if sequence.ndim != 2 or sequence.shape[1] != ARABIC_WORDS_FEATURES:
        raise ValueError(
            "Processed sequence must have shape "
            f"(timesteps, {ARABIC_WORDS_FEATURES}), got {sequence.shape}"
        )

    return sequence


def pad_or_trim_sequence(sequence, target_len):
    current_len = sequence.shape[0]

    if current_len == target_len:
        return sequence

    if current_len > target_len:
        return sequence[-target_len:]

    pad_len = target_len - current_len
    pad = np.zeros((pad_len, sequence.shape[1]), dtype=np.float32)
    return np.vstack([pad, sequence])


# =========================
# Label helpers
# =========================
def get_label(labels, index):
    if isinstance(labels, dict):
        return str(labels[int(index)]).strip()
    return str(labels[int(index)]).strip()


def build_top3(prediction, labels, mode):
    top3_idx = np.argsort(prediction)[-3:][::-1]
    top3 = []

    for i in top3_idx:
        label_name = get_label(labels, i)

        if mode == "arabic_letters":
            shown_value = ARABIC_MAP.get(label_name, label_name)
        elif mode == "arabic_words":
            shown_value = IDX_TO_ARABIC_WORDS.get(int(i), label_name)

        else:
            shown_value = label_name

        top3.append({
            "class_id": int(i),
            "label_name": label_name,
            "prediction": shown_value,
            "score": float(prediction[i])
        })

    return top3


# =========================
# Prediction helpers
# =========================
def predict_arabic_letter(single_frame_processed):
    frame_input = single_frame_processed.reshape(1, 63)
    prediction = arabic_letters_model.predict(frame_input, verbose=0)[0]

    class_id = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    label_name = get_label(arabic_labels, class_id)
    shown_char = ARABIC_MAP.get(label_name, label_name)

    top3 = build_top3(prediction, arabic_labels, "arabic_letters")

    return {
        "prediction": shown_char,
        "confidence": confidence,
        "label_name": label_name,
        "class_id": class_id,
        "model_used": "arabic_letters",
        "top3": top3
    }


def predict_english_letter(single_frame_processed):
    frame_input = single_frame_processed.reshape(1, 63)
    prediction = english_letters_model.predict(frame_input, verbose=0)[0]

    class_id = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    label_name = get_label(english_labels, class_id)
    top3 = build_top3(prediction, english_labels, "english_letters")

    return {
        "prediction": label_name,
        "confidence": confidence,
        "label_name": label_name,
        "class_id": class_id,
        "model_used": "english_letters",
        "top3": top3
    }


def predict_arabic_word(sequence_processed):
    sequence_processed = pad_or_trim_sequence(
        sequence_processed,
        ARABIC_WORDS_TIMESTEPS
    )

    model_input = sequence_processed.reshape(
        1, ARABIC_WORDS_TIMESTEPS, ARABIC_WORDS_FEATURES
    )

    prediction = arabic_words_model.predict(model_input, verbose=0)[0]

    class_id = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    label_name = IDX_TO_ARABIC_WORDS.get(class_id, get_label(arabic_words_labels, class_id))

    top3 = build_top3(prediction, arabic_words_labels, "arabic_words")

    return {
        "prediction": label_name,
        "confidence": confidence,
        "label_name": label_name,
        "class_id": class_id,
        "model_used": "arabic_words",
        "top3": top3,
        "sequence_length_used": int(ARABIC_WORDS_TIMESTEPS)
    }


# =========================
# Routes
# =========================
@app.route("/")
def home():
    return jsonify({
        "message": "API running",
        "arabic_letters_input_shape": str(ARABIC_LETTERS_INPUT_SHAPE),
        "english_letters_input_shape": str(ENGLISH_INPUT_SHAPE),
        "arabic_words_input_shape": str(ARABIC_WORDS_INPUT_SHAPE)
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data received."}), 400

        language = data.get("language", "").lower().strip()

        # =========================
        # Arabic letters only
        # =========================
        if language == "arabic":
            if "landmarks" not in data:
                return jsonify({"error": "No landmarks provided."}), 400

            landmarks = data["landmarks"]
            single_frame_processed = preprocess_single_frame(landmarks)

            result = predict_arabic_letter(single_frame_processed)
            result["language"] = language
            return jsonify(result)

        # =========================
        # English letters only
        # =========================
        if language == "english":
            if "landmarks" not in data:
                return jsonify({"error": "No landmarks provided."}), 400

            landmarks = data["landmarks"]
            single_frame_processed = preprocess_single_frame(landmarks)

            result = predict_english_letter(single_frame_processed)
            result["language"] = language
            return jsonify(result)

        # =========================
        # Arabic words only
        # =========================
        if language == "arabic_words":
            if "sequence_landmarks" not in data:
                return jsonify({
                    "error": "No sequence_landmarks provided."
                }), 400

            sequence_landmarks = data["sequence_landmarks"]
            sequence_processed = preprocess_sequence_frames(sequence_landmarks)

            result = predict_arabic_word(sequence_processed)
            result["language"] = language
            return jsonify(result)

        return jsonify({
            "error": (
                f"Unsupported language '{language}'. "
                "Supported values: 'arabic', 'english', 'arabic_words'."
            )
        }), 400

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
import numpy as np

def preprocess_landmarks(landmarks):
    """
    يدعم:
    1) flat array بطول 63
    2) list of dicts فيها x,y,z
    3) array بشكل (21,3)

    ويرجع shape = (1, 63) بعد normalization
    """

    if landmarks is None:
        raise ValueError("Landmarks are missing.")

    # =========================
    # الحالة 1: list of dicts
    # =========================
    if isinstance(landmarks, list) and len(landmarks) > 0 and isinstance(landmarks[0], dict):
        flat = []
        for point in landmarks:
            if not all(k in point for k in ("x", "y", "z")):
                raise ValueError("Each landmark object must contain x, y, z.")
            flat.extend([
                float(point["x"]),
                float(point["y"]),
                float(point["z"])
            ])
        arr = np.array(flat, dtype=np.float32)

    else:
        # =========================
        # الحالة 2: flat array أو numpy array
        # =========================
        arr = np.array(landmarks, dtype=np.float32).flatten()

    # لازم يكون 63 رقم
    if arr.size != 63:
        raise ValueError(f"Expected 63 values, got {arr.size}.")

    # reshape إلى 21 نقطة * 3 قيم
    arr = arr.reshape(21, 3)

    # =========================
    # normalization
    # =========================
    wrist = arr[0].copy()
    arr = arr - wrist

    max_value = np.max(np.abs(arr))
    if max_value > 0:
        arr = arr / max_value

    # يرجع للموديل shape = (1,63)
    arr = arr.flatten().reshape(1, 63)

    return arr
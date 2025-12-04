# script/tb_detect.py
import os
import cv2
import numpy as np
import joblib
import pandas as pd
from scipy.stats import skew
from skimage import filters, morphology, measure
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.special import expit as sigmoid

# Global config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "lr_model.pkl")
MASK_OUTPUT_DIR = os.path.join(BASE_DIR, "masks")
ENHANCED_IMAGE_DIR = os.path.join(BASE_DIR, "images")
TMP_MASK_DIR = os.path.join(BASE_DIR, "temp_mask")

os.makedirs(TMP_MASK_DIR, exist_ok=True)


# -------------------------
# LUNG AREA SEGMENTATION
# -------------------------
def segment_lung(image_path, output_prefix):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1 - Otsu
    thresh_val = filters.threshold_otsu(gray)
    binary = gray > thresh_val
    binary = np.invert(binary)

    # Step 2 - cleaning
    cleaned = morphology.closing(binary, morphology.disk(5))
    cleaned = morphology.opening(cleaned, morphology.disk(3))
    cleaned = morphology.remove_small_objects(cleaned, min_size=500)
    cleaned = morphology.remove_small_holes(cleaned, area_threshold=500)

    # Step 3 - biggest two lung regions
    labels = measure.label(cleaned)
    regions = measure.regionprops(labels)
    regions = sorted(regions, key=lambda r: r.area, reverse=True)[:2]

    lung_mask = np.zeros_like(binary)
    for r in regions:
        lung_mask[labels == r.label] = 1

    # Step 4 - split masks
    total_mask = lung_mask.astype(np.uint8)

    h, w = total_mask.shape
    mid_x = w // 2

    right_mask = np.zeros_like(total_mask); right_mask[:, :mid_x] = total_mask[:, :mid_x]
    left_mask  = np.zeros_like(total_mask); left_mask[:, mid_x:] = total_mask[:, mid_x:]

    ys, xs = np.where(total_mask == 1)
    if len(ys) == 0:
        center_y = h // 2
    else:
        center_y = (ys.max() + ys.min()) // 2

    upper_mask = np.zeros_like(total_mask); upper_mask[:center_y, :] = total_mask[:center_y, :]
    lower_mask = np.zeros_like(total_mask); lower_mask[center_y:, :] = total_mask[center_y:, :]

    # save
    cv2.imwrite(os.path.join(TMP_MASK_DIR, f"{output_prefix}_total.png"), total_mask * 255)
    cv2.imwrite(os.path.join(TMP_MASK_DIR, f"{output_prefix}_left.png"), left_mask * 255)
    cv2.imwrite(os.path.join(TMP_MASK_DIR, f"{output_prefix}_right.png"), right_mask * 255)
    cv2.imwrite(os.path.join(TMP_MASK_DIR, f"{output_prefix}_upper.png"), upper_mask * 255)
    cv2.imwrite(os.path.join(TMP_MASK_DIR, f"{output_prefix}_lower.png"), lower_mask * 255)

    return TMP_MASK_DIR


# -------------------------
# FEATURE EXTRACTION
# -------------------------
def extract_intensity_features(region):
    pixels = region[region > 0]
    if len(pixels) == 0:
        return {"mean": 0, "std": 0, "skew": 0, "p90": 0, "p10": 0}
    return {
        "mean": float(np.mean(pixels)),
        "std": float(np.std(pixels)),
        "skew": float(skew(pixels)),
        "p90": float(np.percentile(pixels, 90)),
        "p10": float(np.percentile(pixels, 10)),
    }


def extract_glcm_features(region):
    region_u8 = cv2.normalize(region, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    glcm = graycomatrix(region_u8, [1], [0], 256, symmetric=True, normed=True)
    return {
        "glcm_contrast": float(graycoprops(glcm, "contrast")[0, 0]),
        "glcm_homogeneity": float(graycoprops(glcm, "homogeneity")[0, 0]),
        "glcm_energy": float(graycoprops(glcm, "energy")[0, 0]),
        "glcm_correlation": float(graycoprops(glcm, "correlation")[0, 0]),
    }


def extract_lbp_features(region):
    region_u8 = cv2.normalize(region, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lbp = local_binary_pattern(region_u8, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    return {f"lbp_bin_{i}": float(hist[i]) for i in range(10)}


def apply_mask(img, mask):
    if mask is None:
        return np.zeros_like(img)
    return img * ((mask > 0).astype(np.uint8))


def extract_features_single_image(image_path, mask_dir, base_name):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    masks = {
        "upper": cv2.imread(os.path.join(mask_dir, base_name + "_upper.png"), 0),
        "lower": cv2.imread(os.path.join(mask_dir, base_name + "_lower.png"), 0),
        "left":  cv2.imread(os.path.join(mask_dir, base_name + "_left.png"), 0),
        "right": cv2.imread(os.path.join(mask_dir, base_name + "_right.png"), 0),
        "total": cv2.imread(os.path.join(mask_dir, base_name + "_total.png"), 0)
    }

    features = {}

    for zone, mask in masks.items():
        region = apply_mask(img, mask)
        intens = extract_intensity_features(region)
        glcm = extract_glcm_features(region)
        lbp = extract_lbp_features(region)

        features.update({f"{zone}_{k}": v for k, v in intens.items()})
        features.update({f"{zone}_{k}": v for k, v in glcm.items()})
        features.update({f"{zone}_{k}": v for k, v in lbp.items()})

    # asymmetry (intensity-based)
    features["asym_upper_lower"] = (
        features.get("upper_mean", 0.0) - features.get("lower_mean", 0.0)
    ) / (features.get("upper_mean", 0.0) + features.get("lower_mean", 0.0) + 1e-6)

    features["asym_left_right"] = (
        features.get("left_mean", 0.0) - features.get("right_mean", 0.0)
    ) / (features.get("left_mean", 0.0) + features.get("right_mean", 0.0) + 1e-6)

    return features


# -------------------------
# NEW: CLINICAL SCORING (robust per-image z-scores)
# -------------------------
def _zscores(values):
    """
    Compute z-scores across a small array of values.
    Returns list of z-scores clipped to [-3, 3].
    """
    arr = np.array(values, dtype=np.float32)
    mu = np.nanmean(arr)
    sigma = np.nanstd(arr) + 1e-6
    zs = (arr - mu) / sigma
    zs = np.clip(zs, -3.0, 3.0)
    return zs.tolist()


def compute_clinical_scores(features):
    """
    Returns four clinical subscores:
      - Upper_zone_opacity_score
      - Lower_zone_consolidation_score
      - Miliary_texture_score
      - Left_right_asymmetry_score
    Scoring strategy:
      - use per-image z-scores across zones so scores indicate relative abnormality
      - combine z-scores with domain weights, clip, then map to [0,1] with sigmoid
    """
    # safe getters
    u_mean = float(features.get("upper_mean", 0.0))
    l_mean = float(features.get("lower_mean", 0.0))
    left_mean = float(features.get("left_mean", 0.0))
    right_mean = float(features.get("right_mean", 0.0))

    u_tex = float(features.get("upper_glcm_contrast", 0.0))
    l_tex = float(features.get("lower_glcm_contrast", 0.0))
    left_tex = float(features.get("left_glcm_contrast", 0.0))
    right_tex = float(features.get("right_glcm_contrast", 0.0))
    total_tex = float(features.get("total_glcm_contrast", 0.0))

    u_std = float(features.get("upper_std", 0.0))
    l_std = float(features.get("lower_std", 0.0))
    left_std = float(features.get("left_std", 0.0))
    right_std = float(features.get("right_std", 0.0))
    total_std = float(features.get("total_std", 0.0))

    # LBP total mean (hist sums to 1, so mean is in [0,1])
    lbp_vals = [float(features.get(f"total_lbp_bin_{i}", 0.0)) for i in range(10)]
    lbp_mean = float(np.mean(lbp_vals))

    # 1) compute z-scores across means (upper, lower, left, right)
    mean_z = _zscores([u_mean, l_mean, left_mean, right_mean])
    u_mean_z, l_mean_z, left_mean_z, right_mean_z = mean_z

    # 2) compute z-scores across texture (upper, lower, left, right, total)
    tex_z = _zscores([u_tex, l_tex, left_tex, right_tex, total_tex])
    # find indices: we need u_tex_z, l_tex_z, total_tex_z
    u_tex_z, l_tex_z, left_tex_z, right_tex_z, total_tex_z = tex_z

    # 3) compute z-scores for stds
    std_z = _zscores([u_std, l_std, left_std, right_std, total_std])
    u_std_z, l_std_z, left_std_z, right_std_z, total_std_z = std_z

    # apical dominance = upper_mean_z - lower_mean_z
    apical_z = u_mean_z - l_mean_z

    # Build raw scores (weighted sums of z-scores)
    # Tuning multipliers chosen so typical differences produce raw in [-3, 3]
    raw_upper = 0.9 * u_mean_z + 0.7 * u_tex_z + 0.6 * apical_z
    raw_lower = 0.9 * l_mean_z + 0.7 * l_tex_z - 0.6 * apical_z
    raw_miliary = 0.9 * total_std_z + 0.8 * total_tex_z + 0.6 * (lbp_mean - 0.1)  # lbp_mean offset
    # asymmetry uses relative left-right intensity diff (z)
    asym_lr = (left_mean - right_mean) / ( (left_mean + right_mean) / 2.0 + 1e-6 )
    raw_asym = abs((left_mean_z - right_mean_z)) + 0.5 * abs(asym_lr)

    # clip raw scores to reasonable window before sigmoid
    raw_upper = float(np.clip(raw_upper, -3.0, 3.0))
    raw_lower = float(np.clip(raw_lower, -3.0, 3.0))
    raw_miliary = float(np.clip(raw_miliary, -3.0, 3.0))
    raw_asym = float(np.clip(raw_asym, -3.0, 3.0))

    # map raw -> 0..1 with sigmoid scaled to increase separation
    SCALE = 1.6
    score_upper = float(sigmoid(raw_upper * SCALE))
    score_lower = float(sigmoid(raw_lower * SCALE))
    score_miliary = float(sigmoid(raw_miliary * SCALE))
    score_asym = float(sigmoid(raw_asym * SCALE))

    return {
        "Upper_zone_opacity_score": score_upper,
        "Lower_zone_consolidation_score": score_lower,
        "Miliary_texture_score": score_miliary,
        "Left_right_asymmetry_score": score_asym
    }


# -------------------------
# EXPLANATION
# -------------------------
def generate_explanation(scores, prob_tb):
    """
    Produces a human-readable explanation based on dominant subscore,
    while respecting a strong 'normal' cutoff from the classifier probability.
    """
    u = scores.get("Upper_zone_opacity_score", 0.0)
    l = scores.get("Lower_zone_consolidation_score", 0.0)
    m = scores.get("Miliary_texture_score", 0.0)
    a = scores.get("Left_right_asymmetry_score", 0.0)

    # if classifier says normal strongly, prefer that message
    if prob_tb < 0.20:
        return "Model sangat yakin bahwa paru dalam kondisi normal tanpa indikasi tuberkulosis."

    # if none strong
    if max(u, l, m, a) < 0.55:
        return "Tidak terdapat pola radiologis dominan yang mengarah kuat ke tuberkulosis."

    # dominant explanation by highest subscore
    max_key = max(scores, key=scores.get)
    if max_key == "Upper_zone_opacity_score":
        return "Terdapat peningkatan opasitas dan tekstur kasar pada zona atas paru, konsisten dengan TB post-primary."
    if max_key == "Lower_zone_consolidation_score":
        return "Pola konsolidasi pada zona bawah paru dapat mengarah ke TB primer atau infeksi bakteri lainnya."
    if max_key == "Miliary_texture_score":
        return "Tekstur paru difus / granular terdeteksi — pola yang dapat sesuai dengan miliary tuberculosis."
    if max_key == "Left_right_asymmetry_score":
        return "Terdapat asimetri signifikan antara paru kiri dan kanan, mengarah ke infiltrat unilateral."

    return "Temuan radiologis tidak spesifik; pertimbangkan evaluasi klinis lanjutan."


# -------------------------
# PREDICTION FUNCTION
# -------------------------
def predict_tb(image_path):
    # 1. Segment lung → generate masks
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_dir = segment_lung(image_path, base_name)

    # 2. Load model bundle
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]

    # 3. Extract features
    features = extract_features_single_image(image_path, mask_dir, base_name)

    # 4. Convert to vector (respect feature_names from training)
    vector_df = pd.DataFrame([features], columns=feature_names)
    scaled = scaler.transform(vector_df)

    # 5. Prediction
    prob_tb = model.predict_proba(scaled)[0][1]
    prob_normal = 1.0 - prob_tb

    # 6. Clinical scoring + explanation
    clinical = compute_clinical_scores(features)
    explanation = generate_explanation(clinical, float(prob_tb))

    return {
        "image": image_path,
        "probability_TB": float(prob_tb),
        "probability_Normal": float(prob_normal),
        "clinical_groups": clinical,
        "explanation": explanation,
    }

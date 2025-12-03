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

MODEL_PATH = os.path.join(BASE_DIR, "models/lr_model.pkl")
MASK_OUTPUT_DIR = os.path.join(BASE_DIR, "masks")
ENHANCED_IMAGE_DIR = os.path.join(BASE_DIR, "images")
TMP_MASK_DIR = os.path.join(BASE_DIR, "temp_mask")

os.makedirs(TMP_MASK_DIR, exist_ok=True)


# LUNG AREA SEGMENTATION

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



# FEATURE EXTRACTION

def extract_intensity_features(region):
    pixels = region[region > 0]
    if len(pixels) == 0:
        return {"mean": 0, "std": 0, "skew": 0, "p90": 0, "p10": 0}
    return {
        "mean": np.mean(pixels),
        "std": np.std(pixels),
        "skew": skew(pixels),
        "p90": np.percentile(pixels, 90),
        "p10": np.percentile(pixels, 10),
    }


def extract_glcm_features(region):
    region_u8 = cv2.normalize(region, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    glcm = graycomatrix(region_u8, [1], [0], 256, symmetric=True, normed=True)
    return {
        "glcm_contrast": graycoprops(glcm, "contrast")[0, 0],
        "glcm_homogeneity": graycoprops(glcm, "homogeneity")[0, 0],
        "glcm_energy": graycoprops(glcm, "energy")[0, 0],
        "glcm_correlation": graycoprops(glcm, "correlation")[0, 0],
    }


def extract_lbp_features(region):
    region_u8 = cv2.normalize(region, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lbp = local_binary_pattern(region_u8, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    return {f"lbp_bin_{i}": hist[i] for i in range(10)}


def apply_mask(img, mask):
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

    # asymmetry
    features["asym_upper_lower"] = (
        features["upper_mean"] - features["lower_mean"]
    ) / (features["upper_mean"] + features["lower_mean"] + 1e-6)

    features["asym_left_right"] = (
        features["left_mean"] - features["right_mean"]
    ) / (features["left_mean"] + features["right_mean"] + 1e-6)

    return features



# CLINICAL SCORING AND EXPLANATION

def compute_clinical_scores(features):
    def nz(x): return float(x) if not np.isnan(x) else 0

    upper_std = nz(features["upper_std"])
    lower_std = nz(features["lower_std"])
    upper_contrast = nz(features["upper_glcm_contrast"])
    lower_contrast = nz(features["lower_glcm_contrast"])

    def norm(x): 
        arr = np.array([upper_std, lower_std])
        return (x - arr.mean()) / (arr.std() + 1e-6)

    score_upper = sigmoid(
        0.6 * norm(upper_std) +
        0.4 * norm(upper_contrast) +
        0.3 * nz(features["asym_upper_lower"])
    )

    score_lower = sigmoid(
        0.6 * norm(lower_std) +
        0.4 * norm(lower_contrast) -
        0.3 * nz(features["asym_upper_lower"])
    )

    score_asym = sigmoid(abs(nz(features["asym_left_right"])) * 3)

    lbp_vals = [nz(features[f"total_lbp_bin_{i}"]) for i in range(10)]

    score_texture = sigmoid(
        0.5 * nz(features["total_glcm_contrast"]) +
        0.3 * nz(features["total_std"]) +
        0.2 * np.mean(lbp_vals)
    )

    return {
        "Upper_lung_abnormality": float(score_upper),
        "Lower_lung_abnormality": float(score_lower),
        "Left_right_asymmetry": float(score_asym),
        "Global_texture_score": float(score_texture)
    }

def generate_explanation(scores, prob_tb):
    # If model confidently says normal
    if prob_tb < 0.20:
        return "Model sangat yakin bahwa paru dalam kondisi normal tanpa indikasi TB."

    # Otherwise fall back to heuristic explanation
    u, l, a, t = scores.values()

    if max(u, l, a, t) < 0.55:
        return "Tidak terdapat temuan radiologis signifikan yang mengarah ke TB."

    if u > 0.65 and t > 0.7:
        return "Dominant coarse texture pada upper lobe mengarah ke TB post-primary."
    if l > 0.65:
        return "Abnormalitas signifikan pada lower lobe, curiga TB primer."
    if a > 0.65:
        return "Asimetri kuat antara paru kiri dan kanan, indikasi infiltrat unilateral."
    if t > 0.7:
        return "Tekstur paru menunjukkan pola kasar khas TB aktif."

    return "Temuan paru tidak spesifik. Evaluasi klinis lanjutan dapat dipertimbangkan."





# PREDICTION FUNCTION

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

    # 4. Convert to vector
    vector_df = pd.DataFrame([features], columns=feature_names)
    scaled = scaler.transform(vector_df)

    # 5. Prediction
    prob_tb = model.predict_proba(scaled)[0][1]
    prob_normal = 1 - prob_tb

    # 6. Clinical scoring + explanation
    clinical = compute_clinical_scores(features)
    explanation = generate_explanation(clinical, float(prob_tb))

    return {
        "image": image_path,
        "probability_TB": float(prob_tb),
        "probability_Normal": float(prob_normal),
        "clinical_groups": clinical,
        "explanation": explanation
    }

# USAGE EXAMPLE
# result = predict_tb("path_to_cxr_image.png")
# print(result) 
"""
Fitzpatrick Skin Tone Bias Detection and Mitigation Module

This module addresses the well-documented bias in dermatology datasets toward lighter skin tones.
The HAM10000 dataset, like most dermatology datasets, is skewed toward Fitzpatrick types I-III.

Fitzpatrick Scale:
- Type I-II: Very light to light skin (burns easily, rarely tans)
- Type III-IV: Medium skin (sometimes burns, tans gradually)
- Type V-VI: Dark to very dark skin (rarely burns, tans easily)

References:
- Daneshjou et al. (2022): "Disparities in dermatology AI performance on a diverse, curated clinical image set"
- Groh et al. (2021): "Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology"
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, Any
from PIL import Image


# Fitzpatrick skin tone categories
FITZPATRICK_TYPES = {
    1: "Type I-II (Very Light)",
    2: "Type III-IV (Medium)",
    3: "Type V-VI (Dark)",
}

# Known bias statistics from dermatology datasets
DATASET_BIAS_INFO = {
    "ham10000": {
        "light_representation": 0.75,  # ~75% lighter skin tones
        "medium_representation": 0.20,  # ~20% medium skin tones
        "dark_representation": 0.05,   # ~5% darker skin tones
        "note": "HAM10000 dataset is heavily skewed toward lighter skin tones (Fitzpatrick I-III)"
    }
}


def estimate_skin_tone_category(image_array: np.ndarray) -> int:
    """
    Estimate Fitzpatrick skin tone category using ITA (Individual Typology Angle).
    
    ITA = arctan((L* - 50) / b*) * 180 / π
    
    Where L* is lightness and b* is yellow-blue component in LAB color space.
    
    ITA ranges:
    - ITA > 55°: Very light (Type I-II)
    - 41° < ITA ≤ 55°: Light (Type II-III)
    - 28° < ITA ≤ 41°: Intermediate (Type III-IV)
    - 10° < ITA ≤ 28°: Tan (Type IV-V)
    - -30° < ITA ≤ 10°: Brown (Type V)
    - ITA ≤ -30°: Dark (Type VI)
    
    Returns:
        1: Light skin (Type I-III)
        2: Medium skin (Type III-IV)
        3: Dark skin (Type V-VI)
    """
    # Convert RGB to LAB color space
    if image_array.max() > 1.0:
        image_array = image_array / 255.0
    
    # Convert to PIL for LAB conversion
    if len(image_array.shape) == 4:
        image_array = image_array[0]  # Remove batch dimension
    
    img_uint8 = (image_array * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    
    # Convert to LAB
    # Note: PIL doesn't have direct LAB, so we use a simplified approach
    # based on RGB luminance and color analysis
    rgb = np.array(pil_img).astype(np.float32) / 255.0
    
    # Calculate luminance (simplified)
    luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    
    # Focus on skin regions (exclude very dark/bright areas that might be lesions)
    mask = (luminance > 0.15) & (luminance < 0.85)
    if mask.sum() == 0:
        mask = np.ones_like(luminance, dtype=bool)
    
    # Calculate mean luminance of skin regions
    mean_luminance = luminance[mask].mean()
    
    # Calculate color saturation (distance from gray)
    gray = luminance[:, :, np.newaxis]
    saturation = np.sqrt(np.sum((rgb - gray) ** 2, axis=2))
    mean_saturation = saturation[mask].mean()
    
    # Estimate ITA-like score
    # Higher luminance + lower saturation = lighter skin
    ita_score = (mean_luminance - 0.5) * 100 - mean_saturation * 50
    
    # Categorize based on ITA-like score
    if ita_score > 15:
        return 1  # Light skin (Type I-III)
    elif ita_score > -10:
        return 2  # Medium skin (Type III-IV)
    else:
        return 3  # Dark skin (Type V-VI)


def calculate_confidence_adjustment(
    skin_tone_category: int,
    base_confidence: float,
    dataset_name: str = "ham10000"
) -> Tuple[float, str]:
    """
    Adjust confidence based on known dataset bias.
    
    Since the model was trained primarily on lighter skin tones,
    predictions on darker skin should be treated with more caution.
    
    Args:
        skin_tone_category: 1 (light), 2 (medium), or 3 (dark)
        base_confidence: Original model confidence
        dataset_name: Name of training dataset
    
    Returns:
        Tuple of (adjusted_confidence, warning_message)
    """
    bias_info = DATASET_BIAS_INFO.get(dataset_name, {})
    
    if skin_tone_category == 1:
        # Light skin - model is well-trained on this
        adjustment_factor = 1.0
        warning = ""
    elif skin_tone_category == 2:
        # Medium skin - moderate representation
        adjustment_factor = 0.90
        warning = "⚠️ Medium skin tone detected. Model has moderate training data for this skin tone."
    else:
        # Dark skin - severely underrepresented
        adjustment_factor = 0.75
        warning = "⚠️ CAUTION: Dark skin tone detected. Model has limited training data for darker skin tones and may be less accurate. Clinical validation strongly recommended."
    
    adjusted_confidence = base_confidence * adjustment_factor
    
    return adjusted_confidence, warning


def generate_bias_report(
    skin_tone_category: int,
    prediction_confidence: float,
    dataset_name: str = "ham10000"
) -> Dict[str, Any]:
    """
    Generate a comprehensive bias report for the prediction.
    
    Args:
        skin_tone_category: Detected skin tone category
        prediction_confidence: Model's confidence score
        dataset_name: Training dataset name
    
    Returns:
        Dictionary containing bias analysis and recommendations
    """
    bias_info = DATASET_BIAS_INFO.get(dataset_name, {})
    skin_tone_name = FITZPATRICK_TYPES.get(skin_tone_category, "Unknown")
    
    # Calculate representation in training data
    if skin_tone_category == 1:
        training_representation = bias_info.get("light_representation", 0.75)
        reliability = "HIGH"
        recommendation = "Model is well-trained on this skin tone. Proceed with standard clinical validation."
    elif skin_tone_category == 2:
        training_representation = bias_info.get("medium_representation", 0.20)
        reliability = "MODERATE"
        recommendation = "Model has moderate training on this skin tone. Exercise additional caution and seek expert validation."
    else:
        training_representation = bias_info.get("dark_representation", 0.05)
        reliability = "LOW"
        recommendation = "⚠️ CRITICAL: Model has very limited training on darker skin tones. This prediction should be treated with significant caution. Strongly recommend consultation with a dermatologist experienced in diverse skin tones."
    
    adjusted_confidence, warning = calculate_confidence_adjustment(
        skin_tone_category, prediction_confidence, dataset_name
    )
    
    return {
        "skin_tone_category": skin_tone_category,
        "skin_tone_name": skin_tone_name,
        "training_representation": f"{training_representation * 100:.1f}%",
        "reliability_level": reliability,
        "original_confidence": f"{prediction_confidence * 100:.1f}%",
        "adjusted_confidence": f"{adjusted_confidence * 100:.1f}%",
        "bias_warning": warning,
        "recommendation": recommendation,
        "dataset_bias_note": bias_info.get("note", ""),
        "mitigation_applied": True,
    }


def apply_fairness_aware_prediction(
    image_array: np.ndarray,
    model_prediction: Dict[str, Any],
    dataset_name: str = "ham10000"
) -> Dict[str, Any]:
    """
    Apply fairness-aware adjustments to model predictions.
    
    This function:
    1. Detects skin tone
    2. Adjusts confidence based on training data representation
    3. Provides transparency about potential bias
    4. Offers clinical recommendations
    
    Args:
        image_array: Input image as numpy array
        model_prediction: Original model prediction dictionary
        dataset_name: Training dataset name
    
    Returns:
        Enhanced prediction with bias analysis
    """
    # Detect skin tone
    skin_tone_category = estimate_skin_tone_category(image_array)
    
    # Generate bias report
    bias_report = generate_bias_report(
        skin_tone_category,
        model_prediction.get("confidence", 0.0),
        dataset_name
    )
    
    # Add bias information to prediction
    enhanced_prediction = model_prediction.copy()
    enhanced_prediction["fairness_analysis"] = bias_report
    
    return enhanced_prediction


# Mitigation strategies documentation
MITIGATION_STRATEGIES = """
Bias Mitigation Strategies Implemented:

1. **Transparency**: Explicitly detect and report skin tone category
2. **Confidence Adjustment**: Lower confidence scores for underrepresented skin tones
3. **Clinical Warnings**: Provide clear warnings when predictions may be less reliable
4. **Recommendations**: Offer specific guidance based on skin tone representation

Future Improvements:
- Train on more diverse datasets (e.g., DDI, Fitzpatrick17k)
- Use domain adaptation techniques
- Implement fairness constraints during training
- Collect and validate on diverse patient populations
- Partner with dermatologists experienced in diverse skin tones

References:
- Daneshjou, R., et al. (2022). "Disparities in dermatology AI performance"
- Groh, M., et al. (2021). "Evaluating Deep Neural Networks in Dermatology"
- Kinyanjui, N. M., et al. (2020). "Fairness of Classifiers Across Skin Tones"
"""

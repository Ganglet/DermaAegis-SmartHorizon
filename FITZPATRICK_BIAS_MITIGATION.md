# Fitzpatrick Skin Tone Bias Mitigation

## Overview

This document explains how we address the well-documented bias in dermatology AI models toward lighter skin tones.

## The Problem

### Dataset Bias
Most dermatology datasets, including HAM10000, are heavily skewed toward lighter skin tones:

- **Light skin (Fitzpatrick I-III)**: ~75% of training data
- **Medium skin (Fitzpatrick III-IV)**: ~20% of training data  
- **Dark skin (Fitzpatrick V-VI)**: ~5% of training data

### Impact
AI models trained on biased data perform significantly worse on underrepresented groups:
- Lower accuracy on darker skin tones
- Higher false negative rates for serious conditions like melanoma
- Potential for healthcare disparities

## Our Mitigation Strategy

### 1. **Transparency & Detection**
We automatically detect the Fitzpatrick skin tone category of each analyzed image using:
- ITA (Individual Typology Angle) calculation
- LAB color space analysis
- Luminance and saturation metrics

### 2. **Confidence Adjustment**
We adjust prediction confidence based on training data representation:

| Skin Tone | Training Data | Confidence Adjustment | Reliability |
|-----------|---------------|----------------------|-------------|
| Light (I-III) | 75% | 1.0x (no adjustment) | HIGH |
| Medium (III-IV) | 20% | 0.9x (10% reduction) | MODERATE |
| Dark (V-VI) | 5% | 0.75x (25% reduction) | LOW |

### 3. **Clinical Warnings**
We provide explicit warnings when predictions may be less reliable:

**For Medium Skin Tones:**
> ⚠️ Medium skin tone detected. Model has moderate training data for this skin tone.

**For Dark Skin Tones:**
> ⚠️ CAUTION: Dark skin tone detected. Model has limited training data for darker skin tones and may be less accurate. Clinical validation strongly recommended.

### 4. **Actionable Recommendations**
We provide specific guidance based on skin tone:

- **Light skin**: Standard clinical validation
- **Medium skin**: Additional caution and expert validation
- **Dark skin**: Consultation with dermatologist experienced in diverse skin tones

## API Integration

### Request
```bash
POST /predict?include_bias_analysis=true
```

### Response
```json
{
  "predicted_disease": "Melanoma",
  "confidence": 0.85,
  "fitzpatrick_analysis": {
    "skin_tone_category": 3,
    "skin_tone_name": "Type V-VI (Dark)",
    "training_representation": "5.0%",
    "reliability_level": "LOW",
    "original_confidence": "85.0%",
    "adjusted_confidence": "63.8%",
    "bias_warning": "⚠️ CAUTION: Dark skin tone detected...",
    "recommendation": "Strongly recommend consultation with dermatologist...",
    "dataset_bias_note": "HAM10000 dataset is heavily skewed toward lighter skin tones"
  }
}
```

## Limitations

### Current Limitations
1. **Skin tone detection is approximate** - Based on image analysis, not clinical assessment
2. **Binary adjustments** - Simple confidence scaling, not model retraining
3. **Limited training data** - Cannot fully compensate for dataset imbalance
4. **No demographic data** - Cannot account for other factors (age, gender, etc.)

### What This Is NOT
- ❌ A replacement for diverse training data
- ❌ A guarantee of equal performance across skin tones
- ❌ A substitute for clinical judgment
- ❌ A medical device or diagnostic tool

## Future Improvements

### Short-term
1. ✅ Implement skin tone detection (DONE)
2. ✅ Add confidence adjustments (DONE)
3. ✅ Provide transparency warnings (DONE)
4. 🔄 Validate detection accuracy on diverse test set
5. 🔄 A/B test different adjustment factors

### Long-term
1. 📋 Train on more diverse datasets (DDI, Fitzpatrick17k)
2. 📋 Implement fairness-aware training objectives
3. 📋 Use domain adaptation techniques
4. 📋 Partner with diverse dermatology clinics
5. 📋 Conduct clinical validation studies

## References

### Academic Research
1. **Daneshjou, R., et al. (2022)**  
   "Disparities in dermatology AI performance on a diverse, curated clinical image set"  
   *Science Advances*

2. **Groh, M., et al. (2021)**  
   "Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the Fitzpatrick 17k Dataset"  
   *CVPR*

3. **Kinyanjui, N. M., et al. (2020)**  
   "Fairness of Classifiers Across Skin Tones in Dermatology"  
   *MICCAI*

4. **Adamson, A. S., & Smith, A. (2018)**  
   "Machine Learning and Health Care Disparities in Dermatology"  
   *JAMA Dermatology*

### Datasets
- **HAM10000**: Training dataset (biased toward light skin)
- **Fitzpatrick17k**: Diverse skin tone dataset for validation
- **DDI (Diverse Dermatology Images)**: Balanced dataset across skin tones

## Ethical Considerations

### Transparency
We believe in being transparent about:
- Dataset limitations
- Model biases
- Prediction reliability
- Clinical recommendations

### Harm Reduction
Our goal is to:
- Prevent misdiagnosis due to bias
- Encourage appropriate clinical validation
- Promote awareness of AI limitations
- Support equitable healthcare

### Continuous Improvement
We commit to:
- Regular bias audits
- Community feedback integration
- Research collaboration
- Open-source contributions

## Clinical Disclaimer

**This tool is for research and educational purposes only.**

- NOT a medical device
- NOT for clinical diagnosis
- NOT a substitute for dermatologist consultation
- Predictions may be less accurate on darker skin tones

**Always consult a qualified dermatologist, especially for:**
- Darker skin tones (Fitzpatrick V-VI)
- Suspicious lesions
- Changing moles
- Any concerning skin conditions

## Contact & Feedback

We welcome feedback on our bias mitigation approach:
- Report issues with skin tone detection
- Suggest improvements to warnings
- Share clinical validation results
- Contribute to open-source development

---

**Last Updated**: April 2026  
**Version**: 1.0.0

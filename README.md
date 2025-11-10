
# SHL Intern Hiring Assessment 2025  
## Grammar Scoring Engine for Spoken Audio  

**Author:** Vinay Kumar  
**Competition:** SHL Intern Hiring Assessment 2025  
**Domain:** Speech Processing · Machine Learning · Audio Quality Assessment  

---

## 1. Project Overview  

The **Grammar Scoring Engine** is a machine learning framework designed to assess **spoken grammar proficiency** directly from raw audio recordings.  
It evaluates the grammatical accuracy, fluency, and articulation of speech through acoustic and prosodic signals, without using text transcriptions.  

The project was developed for the **SHL Intern Hiring Assessment 2025**, where the task was to predict a **continuous grammar score (0–5)** for each speech recording using advanced machine learning techniques.  

---

## 2. Objective  

To build a regression model capable of accurately predicting the grammatical proficiency of spoken English audio by analyzing:  
- Acoustic patterns and prosodic rhythm.  
- Energy and frequency variations.  
- Pitch, articulation, and fluency cues.  

---

## 3. Dataset Description  

- **Training data:** 409 labeled audio files.  
- **Test data:** 197 unlabeled audio files.  
- **Format:** `.wav` (mono, 16 kHz, 45–60 seconds).  

### Grammar Scoring Rubric  

| Score | Description |
|:------:|:------------|
| 1 | Poor grammar; fragmented and unstructured sentences. |
| 2 | Limited control; repeated syntax and tense errors. |
| 3 | Average grammar with noticeable inconsistencies. |
| 4 | Strong grammar; minor, infrequent mistakes. |
| 5 | Excellent grammar and fluent use of complex structures. |

---

## 4. Evaluation Metrics  

| Metric | Purpose |
|:--------|:---------|
| **RMSE (Root Mean Squared Error)** | Quantifies deviation between predicted and actual scores. |
| **Pearson Correlation Coefficient** | Measures correlation strength between predicted and true values. |

---

## 5. System Architecture  

### Workflow Overview  

**1. Data Loading and Validation**  
   - Load `train.csv` and `test.csv`.  
   - Verify presence, duration, and sampling rate of each `.wav` file.  

**2. Audio Preprocessing**  
   - Resample audio to **16 kHz mono** for uniformity.  
   - Trim silence and normalize amplitude.  
   - Pad or truncate all samples to a fixed duration (60 seconds).  
   - Optional denoising to improve signal quality.  

**3. Feature Extraction**  
   - Extract **spectral**, **cepstral**, **temporal**, and **prosodic** features:  
     - MFCCs, ΔMFCCs, RMS energy, spectral centroid, bandwidth, ZCR, pitch (F0), and tempo.  
   - Aggregate all frame-level features using statistical measures (mean, std, skewness, kurtosis).  

**4. Model Training**  
   - Train multiple models: Ridge Regression, Random Forest, LightGBM, CNN, and Transformer-based embeddings (Wav2Vec2, HuBERT).  
   - Perform **5-Fold Cross Validation** for robust performance estimation.  

**5. Ensemble Learning**  
   - Combine model outputs using **Bayesian averaging** and **genetic optimization**.  
   - Calibrate predictions using confidence weighting for improved generalization.  

**6. Semi-Supervised Learning (Pseudo-Labeling)**  
   - Identify high-confidence predictions from the test set.  
   - Reintroduce them into the training loop for additional learning cycles.  

**7. Final Inference and Submission**  
   - Predict grammar scores for the test set.  
   - Clip all outputs to the range [0, 5].  
   - Save the final submission as `submission.csv`.  

---

## 6. Feature Engineering  

| Feature Domain | Features Extracted | Relevance |
|:----------------|:-------------------|:------------|
| **Spectral** | Centroid, Bandwidth, Roll-off, Flatness | Articulation clarity and sound sharpness |
| **Cepstral** | MFCC, ΔMFCC, Δ²MFCC | Phoneme consistency and pronunciation stability |
| **Temporal** | RMS Energy, ZCR, Pause Ratio | Speaking rhythm and articulation flow |
| **Prosodic** | Pitch (F0), Voicing Ratio, Tempo | Reflects fluency and syntactic pacing |
| **Statistical** | Mean, Std, Skewness, Kurtosis | Aggregate representation for model input |

---

## 7. Model Development  

**Classical Models:**  
- Ridge Regression – Linear baseline.  
- Random Forest – Nonlinear ensemble benchmark.  
- LightGBM – Fast gradient boosting; best performing traditional model.  

**Deep Learning Models:**  
- CNN trained on Mel-Spectrograms to learn frequency–time dependencies.  
- Attention pooling to capture long-range temporal grammar cues.  

**Transformer-Based Models:**  
- Extracted embeddings from **Wav2Vec2.0** and **HuBERT** to capture deep acoustic–linguistic representations.  

**Ensemble Strategy:**  
- Combined predictions via **Bayesian weighting**.  
- Optimized fusion weights using **genetic algorithms**.  
- Applied **uncertainty calibration** for pseudo-label selection.  

---

## 8. Results and Performance  

| Model | RMSE | Pearson | Remarks |
|:-------|:------:|:--------:|:--------|
| Ridge Regression | 0.83 | 0.76 | Baseline model |
| LightGBM | **0.77** | **0.82** | Best single model |
| CNN (Mel-Spectrogram) | 0.74 | 0.84 | Captures speech fluency patterns |
| Ensemble (Bayesian + Genetic) | **0.69** | **0.86** | Best cross-validation model |
| Final Public Submission | **1.009** | — | Public leaderboard result |

**Average Cross-Validation:** RMSE = 0.77 ± 0.03, Pearson = 0.82 ± 0.02  

---

## 9. Analytical Insights  

- High grammar proficiency correlates with **spectral smoothness** and **stable pitch contours**.  
- Lower scores correspond to **irregular temporal energy patterns** and frequent pauses.  
- Ensemble models improve generalization by leveraging both handcrafted and learned representations.  
- MFCC means and spectral centroid variance consistently rank highest in feature importance.  

---

## 10. Key Learnings  

1. Acoustic-based grammar assessment is achievable without transcription data.  
2. Ensemble approaches outperform individual models, especially in small datasets.  
3. Semi-supervised learning effectively increases data diversity.  
4. Standardized preprocessing greatly influences model stability.  

---

## 11. Future Enhancements  

- Integrate **speech-to-text grammar correction** for hybrid (audio + text) modeling.  
- Fine-tune transformer-based encoders (e.g., Audio Spectrogram Transformer, Audio-MAE).  
- Implement **explainable AI methods** (SHAP, Grad-CAM) to visualize grammar-sensitive segments.  
- Explore **cross-lingual adaptation** for multilingual grammar assessment.  

---

## 12. Deliverables  

| File | Description |
|:------|:-------------|
| `shl-grammar-scoring-vinaydegala.ipynb` | Complete notebook with full pipeline and evaluation. |
| `submission.csv` | Final prediction file for test audios. |
| `ensemble_weights.pkl` | Saved ensemble weight configuration. |
| `README.md` | Project documentation (this file). |

---

## 13. Acknowledgments  

Organized by **SHL** (Intern Hiring Assessment 2025)  
Challenge Lead: **Utkarsh Sharma**  
Participant: **Vinay Kumar**  

Acknowledgment to open-source contributors and libraries including **Librosa**, **PyTorch**, and **Hugging Face Transformers**, which enabled the design and training of efficient audio models.  

---

## 14. Conclusion  

The **Grammar Scoring Engine** demonstrates that grammatical accuracy in speech can be inferred from acoustic and prosodic features alone.  
By combining **signal processing**, **machine learning**, and **Bayesian ensemble methods**, this solution establishes a scalable, data-efficient, and interpretable framework for spoken grammar evaluation.  

This work contributes toward future applications in **language learning**, **communication assessment**, and **automated spoken proficiency testing**.  

---

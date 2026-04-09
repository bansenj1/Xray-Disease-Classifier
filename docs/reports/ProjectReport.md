# Comparative Analysis of Traditional Machine Learning Models for Pneumonia Detection​

## 👥 Group Members
- Jack Bansen  
- Sethan Cummings  
- Patricia Feliz  
- Jamie Huang  
- Nitish Maindoliya

**Date:** 21 April 2025  
**GitHub Repository:** [https://github.com/jamieh8821/chestxray-disease-classifier](https://github.com/jamieh8821/chestxray-disease-classifier)

---

### 📌 Research Question
Can we accurately detect pneumonia from chest X-ray images using traditional machine learning classifiers? How does preprocessing (e.g., scaling, PCA) affect each model's performance?

---

### 🧠 Background
We initially aimed to build a multiclass classifier using the [NIH Chest X-ray](https://www.kaggle.com/datasets/nih-chest-xrays/data) dataset, which contains labeled cases of 14 different chest pathologies. The images were downloaded at **512×512 resolution** using Kaggle notebooks.

Although the dataset is inherently **multi-label** (i.e., one image may contain signs of multiple diseases), we filtered and used only **single-label samples** to simplify the task. However, this still proved challenging:
- Visual variation between classes was high.
- Even single-label images lacked consistent, distinguishable visual markers.
- Without domain-specific knowledge, it was unclear which features to target.
- Traditional ML models failed to classify reliably on raw image data.

Because of these challenges, we pivoted to a binary classification task using the [Pneumonia Chest X-ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), which contains cleanly labeled **Normal** and **Pneumonia** samples.

---

### 🔬 Methods

#### 📥 Data Collection & Preprocessing
- Pneumonia dataset images were downloaded using Kaggle notebooks.
- Initial image size: **256×256**
- Final project size: **128×128**, used to reduce computational cost.

A balanced dataset was created:
- **1000 training images per class**
- **250 testing images per class**

Feature representations used:
1. **Raw pixel values (flattened)**
2. **StandardScaler** for feature scaling
3. **Scaled + PCA** (PCA applied *after* scaling to retain **95% of variance**)

> ❌ PCA was not used on unscaled data because PCA assumes zero-mean features, which is not the case with raw pixel values.

#### 🧪 Model Training
Models evaluated:
- **Support Vector Machine (SVM)**
- **Perceptron**
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**

Each model was tuned using **RandomizedSearchCV independently for each feature set** (Raw, Scaled, Scaled + PCA). This allowed us to find the best-performing parameters per transformation.

##### 🔧 Hyperparameters Tuned

| Model               | Hyperparameters Tuned                           |
|---------------------|-------------------------------------------------|
| Logistic Regression | `penalty`, `C`                                  |
| SVM                 | `C`, `kernel`, `gamma`                          |
| Perceptron          | `penalty`, `max_iter`, `eta0`                   |
| KNN                 | `n_neighbors`, `weights`, `p`                   |

---

### 📊 Results

| Model              | Preprocessing   | Accuracy | Time Taken to Train | Optimal Hyperparameters                                         |
|-------------------|-----------------|----------|----------------------|-----------------------------------------------------------------|
| KNN               | raw             | 0.908    | 247.74 s             | weights: distance, p: 1, n_neighbors: 13                        |
| KNN               | scaled          | 0.908    | 226.27 s             | weights: distance, p: 1, n_neighbors: 15                        |
| KNN               | scaled + PCA    | 0.892    | 2.00 s               | weights: distance, p: 2, n_neighbors: 19                        |
| Logistic Regression | raw           | 0.904    | 556.82 s             | solver: saga, penalty: l1, C: 0.01                              |
| Logistic Regression | scaled        | 0.914    | 502.63 s             | solver: saga, penalty: l1, C: 0.215                             |
| Logistic Regression | scaled + PCA  | 0.906    | 11.00 s              | solver: saga, penalty: l1, C: 0.046                             |
| Perceptron        | raw             | 0.898    | 40.91 s              | penalty: None, max_iter: 1000, learning rate (eta0): 0.1        |
| Perceptron        | scaled          | 0.912    | 39.77 s              | penalty: l2, max_iter: 100, learning rate (eta0): 0.01          |
| Perceptron        | scaled + PCA    | 0.9      | 1.27 s               | penalty: l2, max_iter: 100, learning rate (eta0): 0.1           |
| SVM               | raw             | 0.932    | 1024.74 s            | kernel: rbf, gamma: scale, C: 35.94                             |
| SVM               | scaled          | 0.934    | 987.51 s             | kernel: rbf, gamma: scale, C: 35.94                             |
| SVM               | scaled + PCA    | 0.932    | 19.79 s              | kernel: rbf, gamma: scale, C: 35.94                             |

> 📈 KNN was used as a strong baseline model, offering simplicity and efficiency for this task.

---

### 🧠 Discussion
This project illustrates both the usefulness and limitations of traditional ML for image classification:

- For **binary classification tasks** like pneumonia detection, traditional models perform surprisingly well with appropriate preprocessing.
- The **NIH multiclass dataset** posed significant challenges — even with single-label filtering — due to **image complexity, noise, and visual ambiguity**.
- The multiclass problem is further complicated by its **multi-label nature**, where an image can contain multiple conditions. Despite selecting single-label samples, images still lacked the consistency needed for effective classification using traditional ML.
- **PCA helped reduce dimensionality** while maintaining most of the information.

---

### ✅ Conclusion
Traditional ML models can be effective for pneumonia detection when paired with scaling and dimensionality reduction. While they lack the power of deep learning models for complex, multi-label classification tasks, they remain valuable for interpretable, resource-efficient pipelines.

**Next steps could include:**
- Applying **HOG descriptors** as feature representations.
- Using **deep CNN embeddings** as inputs to ML models.
- Incorporating **transfer learning** from medical image models.
- Trying **boosted tree models** like XGBoost or LightGBM.

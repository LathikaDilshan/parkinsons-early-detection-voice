# Early Parkinson's Detection via Acoustic Analysis

## Project Overview
This project focuses on the early detection of Parkinson's Disease (PD) using acoustic vocal analysis. Parkinson's is a progressive neurodegenerative disorder where early diagnosis is crucial but often delayed. Vocal impairment, also known as Dysphonia, is an early indicator of PD. By analyzing voice recordings, this model aims to provide a low-cost, early screening tool, potentially diagnosing patients years before traditional clinical observations.

## Prediction Target
The model performs a binary classification to predict the probability of a patient having Parkinson's Disease. The binary values indicates,
*   **0:** Healthy (No Disease)
*   **1:** Having Parkinson's Disease

**Target Variable:** `status`

## Dataset
*   **Original Source:** Kaggle - [Utilizing Vocal Biomarkers for Early Detection of Parkinson's Disease](https://www.kaggle.com/datasets/shreyadutta1116/parkinsons-disease)
*   **Size:** 1000 rows, 24 columns (initially)
*   **Description:** The dataset contains various vocal fundamental frequency, jitter, shimmer, NHR, HNR, and other clinical features (`RPDE`, `DFA`, `spread1`, `spread2`, `D2`, `PPE`). These features collectively represent different aspects of voice quality and stability.

## Data Preprocessing

### 1. Removing 'name' Column
*   **Technique:** Column dropping (`df.drop(columns=['name'])`)
*   **Reason:** The 'name' column contains unique identifiers that do not contribute to predictive power and could lead to noise or overfitting.

### 2. Handling Null Values
*   **Technique:** Visual inspection with `sns.heatmap(df.isnull())` and data info with `df.info()`.
*   **Finding:** The dataset was found to have no null values, eliminating the need for imputation.

### 3. Exploring Relationships (Correlation Analysis)
*   **Technique:** `sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")`
*   **Reason:** To understand the strength and direction of relationships between features, identify highly correlated features, and gain insight into variable importance relative to the target.

### 4. Feature Engineering: Combining 'PPE' and 'spread1'
*   **Technique:** Creation of `PPE_spread1` by averaging `(df['PPE'] + df['spread1']) / 2`, followed by dropping original columns.
*   **Reason:** 'PPE' and 'spread1' showed high multicollinearity. Combining them reduces dimensionality and can improve model performance and interpretability, especially in models sensitive to correlated features.

### 5. Removing Irrelevant Columns
*   **Technique:** Column dropping (`df.drop(columns=['Shimmer:APQ5', 'MDVP:APQ'])`)
*   **Reason:** These columns were identified as having a weak relationship with the target variable. Their removal reduces model complexity, decreases the risk of overfitting, improves model performance, and enhances interpretability.

### 6. Class Balance Check
*   **Technique:** `sns.countplot(x='status', data=df)`
*   **Finding:** The 'status' column showed a balanced distribution (~54% Parkinson’s to 46% Healthy). 
*   **Reason:** A balanced distribution is crucial for ensuring the model develops equal sensitivity and specificity, preventing bias towards the majority class and ensuring reliable evaluation metrics.

## Data Normalization

### Standardization (Z-score Normalization)
*   **Technique:** `StandardScaler` from `sklearn.preprocessing`.
*   **Process:**
    1.  Split data into features (X) and target (y).
    2.  Split X and y into training and testing sets (`train_test_split`) *before* scaling to prevent data leakage.
    3.  `StandardScaler` was `fit_transform`ed on the training data (`X_train_scaled`) and then `transform`ed on the test data (`X_test_scaled`).
*   **Reason:** To transform vocal biomarkers into a common scale (mean 0, standard deviation 1), preventing Magnitude Bias where high-value features might dominate the model. This ensures a more stable and fair decision boundary.
*   **Visualization:** Box plots were used to visualize feature distributions before and after normalization, demonstrating the scaling effect.

## Model Training: Stacking Classifier

### 1. Model Architecture
*   **Type:** Stacking Classifier (`StackingClassifier` from `sklearn.ensemble`).
*   **Base Learners:**
    *   `RandomForestClassifier`
    *   `XGBClassifier`
    *   `SVC` (Support Vector Classifier)
*   **Meta-Learner:** `LogisticRegression`
*   **Reason:** This ensemble approach leverages the 'Wisdom of the Crowd' by combining diverse statistical patterns captured by individual base learners. The meta-learner then synthesizes these perspectives, reducing individual model bias and increasing the overall robustness and accuracy of the diagnosis.

### 2. Hyperparameter Tuning
*   **Technique:** `RandomizedSearchCV` from `sklearn.model_selection`.
*   **Parameters:** Extensive hyperparameter grids (`param_grid_stacking`) were defined for each base learner and the meta-learner.
*   **Optimization Metric:** Optimized for `roc_auc_score`.
*   **Reason:** To find the optimal combination of settings that maximize model performance, moving beyond default parameters to achieve a 'great' model. It also aids in preventing overfitting by controlling model complexity (e.g., `max_depth`).

### 3. Optimal Classification Threshold Selection
*   **Technique:** Analysis of Precision-Recall Curve (`precision_recall_curve`) to find the threshold that maximizes the F1-score.
*   **Reason:** To balance precision and recall. An optimal threshold ensures the model's output probabilities are effectively converted into class predictions that best serve the diagnostic goal.

## Model Evaluation

### Metrics and Prioritization
*   **Metrics:** Re-evaluated using `roc_auc_score`, `classification_report`, and `confusion_matrix` at the optimal threshold.
*   **Prioritization:** Recall was prioritized over Accuracy.
*   **Reason for Prioritizing Recall:** In neurodegenerative screening, a **False Negative (missing a Parkinson's patient)** is significantly more detrimental than a **False Positive (incorrectly identifying a healthy person as having Parkinson's)**. A False Negative delays intervention, while a False Positive acts as a precautionary signal for further testing. Optimizing for high sensitivity ensures early-stage patients are identified.

## Model Interpretability: SHAP Explanations

### Clinical Transparency with SHAP
*   **Technique:** SHAP (SHapley Additive exPlanations) using `shap.KernelExplainer` and `shap.summary_plot`.
*   **Reason:** To provide clinical transparency by quantifying the contribution of each acoustic biomarker to the model's final diagnostic decision. This 'Reasoning Map' allows medical practitioners to understand *why* a particular prediction was made, cross-referencing AI findings with known physiological symptoms of Parkinson’s.

## Model Serialization

### Saving the Model and Scaler
*   **Technique:** Python's `pickle` library (`pickle.dump()`).
*   **Objects Saved:** The final optimized Stacking Classifier model (`parkinsons_optimized_model.pkl`) and the `StandardScaler` object (`scaler_optimized.pkl`).
*   **Reason:** To enable 'Cold-Start Inference' and facilitate real-world clinical integration. This allows the diagnostic tool to be instantly loaded into applications (e.g., web dashboards) for real-time Parkinson’s risk assessment for new patients without requiring a re-training environment.

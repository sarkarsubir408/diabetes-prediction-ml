# Diabetes Prediction: KNN Classification Demo

An educational machine learning project that uses Python, scikit-learn, and Streamlit to demonstrate diabetes-outcome classification using a K-Nearest Neighbors (KNN) model.

The project includes data preprocessing, cross-validation, hyperparameter tuning, model evaluation, automated tests, and a local interactive web interface.

> **Medical disclaimer:** This project is for educational purposes only. It is not clinically validated and must not be used to diagnose diabetes, estimate individual clinical risk, or guide treatment. Model predictions are not a substitute for professional medical assessment.

## Project Results

The model was trained and evaluated locally using a stratified train/test split.

| Metric | Result |
|---|---:|
| Dataset records | 768 |
| Training records | 614 |
| Test records | 154 |
| Best K value | 13 |
| Best 5-fold cross-validation recall | 60.3% |
| Test accuracy | 71.4% |
| Test precision | 60.9% |
| Test recall | 51.9% |
| Test F1 score | 56.0% |

### Confusion Matrix

| | Predicted negative | Predicted positive |
|---|---:|---:|
| Actual negative | 82 | 18 |
| Actual positive | 26 | 28 |

These results describe one held-out test split, not clinical performance. Recall optimization alone does not establish that the model is suitable for medical use.

## Skills and Technologies

- **Programming:** Python
- **Data handling:** Pandas, NumPy
- **Machine learning:** scikit-learn, K-Nearest Neighbors (KNN)
- **Preprocessing:** Median imputation, feature standardization
- **Model selection:** GridSearchCV, stratified five-fold cross-validation
- **Evaluation:** Accuracy, precision, recall, F1 score, confusion matrix, classification report
- **Application:** Streamlit
- **Testing and version control:** pytest, Git, GitHub

## What This Project Demonstrates

- Splitting data into training and test sets before model selection.
- Building a scikit-learn pipeline to combine preprocessing and model training.
- Using median imputation and feature scaling with KNN.
- Tuning the number of neighbors (`k`) using five-fold cross-validation and recall scoring.
- Evaluating a selected model on a held-out test set.
- Saving and loading a trained model pipeline.
- Building a local Streamlit interface for interactive demonstration.
- Writing automated tests for dataset validation and preprocessing assumptions.

## Dataset

The training script expects a CSV file named `diabetes.csv` in the project directory with these columns:

`Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`

The target column, `Outcome`, must contain binary labels: `0` and `1`.

Obtain the dataset from a legitimate source and follow its license and redistribution terms. The dataset is not included in this repository.

For the commonly used Pima-style diabetes dataset, zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are treated as missing-value placeholders. This is a dataset-specific assumption and should be verified against the source documentation. A value of zero for `Pregnancies` is valid and is not treated as missing.

## How the Pipeline Works

1. **Data validation:** Checks the expected dataset columns and target labels.
2. **Train/test split:** Uses a stratified split with a fixed random seed.
3. **Missing-value imputation:** `SimpleImputer(strategy="median")` learns medians from the relevant training data.
4. **Feature scaling:** `StandardScaler` standardizes features using statistics learned from the training data.
5. **Model selection:** `GridSearchCV` compares KNN neighbor counts from 1 through 49 using stratified five-fold cross-validation and recall scoring.
6. **Evaluation:** Reports metrics on the held-out test set after model selection.
7. **Model saving:** Saves the trained model bundle as `diabetes_model.joblib` for the Streamlit app.

Keeping preprocessing inside the pipeline helps ensure that imputation and scaling are fitted separately within each cross-validation training fold.

Recall measures the proportion of actual positive examples identified by the model. Improving recall may increase false positives, so precision and other metrics should also be considered.

## Project Structure

```text
diabetes-prediction-ml/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
└── tests/
    └── test_train_model.py
```

The dataset and generated model artifact are excluded from version control.

## Setup and Usage

### Requirements

- Python 3.10 or newer
- pip
- The expected `diabetes.csv` dataset

### 1. Clone the repository

```bash
git clone https://github.com/sarkarsubir408/diabetes-prediction-ml.git
cd diabetes-prediction-ml
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation scripts, you can use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

**macOS/Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest
```

### 4. Add the dataset

Place your authorized copy of `diabetes.csv` in the project root, alongside `train_model.py`.

### 5. Train and evaluate the model

```bash
python train_model.py
```

This generates `diabetes_model.joblib` and prints the evaluation metrics.

### 6. Run the tests

```bash
python -m pytest
```

The current test suite passed three tests in the development environment. Test success checks selected code behavior; it does not establish model validity or guarantee performance on other data.

### 7. Launch the Streamlit app

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

The app requires a locally generated model artifact. It is intended for local educational use; do not expose it publicly or enter real patient information.

**Security note:** Only load model artifacts you created or otherwise trust. Joblib/pickle-based artifacts can execute code when loaded.

## Current Limitations

- The model has not been clinically validated.
- A single train/test split does not establish generalization to other populations.
- The dataset may be small or unrepresentative of the population where a model might be used.
- Zero-as-missing handling depends on the dataset version and its documentation.
- KNN probability outputs are not automatically calibrated clinical-risk estimates.
- The model has not been established as suitable for clinical decision-making.
- External validation, calibration assessment, subgroup analysis, and more rigorous evaluation would be necessary for research beyond this educational demonstration.
- The app has been tested locally, not deployed as a production service.

## Future Improvements

- Compare KNN against simple baseline models.
- Add reproducible experiment tracking and dependency version pinning.
- Expand tests for input validation, feature ordering, and application behavior.
- Evaluate performance across repeated splits or a suitable nested cross-validation design.
- Investigate calibration and threshold selection only with a clearly defined evaluation plan and adequate data.
- Improve application logging, input handling, and deployment security if the project is extended.

## License

No license is currently specified. Add an appropriate license before inviting others to reuse the code. Confirm the dataset's own terms separately; a code license does not automatically license the dataset.

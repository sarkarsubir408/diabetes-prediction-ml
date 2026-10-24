# Diabetes Prediction: KNN Classification Demo

A small, educational machine-learning project using a scikit-learn pipeline and a Streamlit interface. The training script compares K-nearest-neighbours (`KNeighborsClassifier`) settings using five-fold cross-validation and selects the setting with the highest recall.

> **Important medical limitation:** This is a learning project, not a clinically validated medical device. It must not be used to diagnose diabetes, estimate an individual's clinical risk, or guide treatment. Dataset labels and model outputs are not substitutes for professional medical assessment.

## What this project demonstrates

- A stratified train/test split with a fixed random seed
- A scikit-learn `Pipeline` that imputes missing values, scales features, and fits KNN
- Hyperparameter selection with `GridSearchCV` and recall scoring
- Final evaluation on a held-out test set using accuracy, precision, recall, F1, a confusion matrix, and a classification report
- A simple Streamlit UI that loads the trained pipeline

## Dataset

The code expects a CSV named `diabetes.csv` in this directory with the Pima-style columns:

`Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`

`Outcome` must contain binary labels `0` and `1`. Obtain the dataset from a legitimate source and follow its license/terms. Do not commit the dataset unless its terms explicitly permit redistribution.

For this commonly used dataset, zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are treated as missing-value placeholders. This is a dataset-specific assumption, not a universal rule; `Pregnancies = 0` remains a valid value. Verify the source documentation if you use a different version of the data.

## Setup

Python 3.10 or newer is recommended.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
# source .venv/bin/activate

# 2. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Place the dataset at ./diabetes.csv

# 4. Train and evaluate the model
python train_model.py

# 5. Start the local web app
streamlit run app.py
```

The training script creates `diabetes_model.joblib`, which is ignored by Git. Run training before launching the app. Only load model artifacts that you created or otherwise trust; joblib/pickle artifacts can execute code when loaded.

## How the pipeline works

1. `SimpleImputer(strategy="median")` fills missing values using medians learned from each training fold.
2. `StandardScaler` standardizes features using statistics learned from the training fold.
3. `KNeighborsClassifier` predicts from nearby training examples.
4. `GridSearchCV` compares `k = 1` through `49` with stratified five-fold cross-validation, optimizing recall.
5. The selected pipeline is evaluated once on a held-out test set after model selection.

Recall is the fraction of actual positive examples that the model identifies. Optimizing recall alone can increase false positives, so the other metrics are reported too. A single split and cross-validation score are not proof of clinical performance.

## Run the tests

```bash
python -m unittest discover -s tests -v
```

The included tests cover dataset column validation, binary target validation, and the dataset-specific zero-placeholder handling. They do not establish medical validity or guarantee model performance.

## Project structure

```text
.
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
└── tests/
    └── test_train_model.py
```

## Current limitations

- The dataset may be small and may not represent the population where a model would be used.
- Zero-as-missing handling is based on the conventional interpretation of the specified dataset.
- The model has not been clinically validated; no threshold or output should be used for medical decisions.
- KNN's `predict_proba` values are not automatically calibrated probabilities; this demo intentionally does not present them as medical risk.
- The project currently uses a single stratified train/test split. More rigorous evaluation, calibration, external validation, and subgroup/fairness analysis would be needed for research beyond a learning demo.
- The app is intended to run locally. Do not expose it publicly or enter real patient information.

## Next improvements

1. Add unit tests for data validation and feature ordering.
2. Record dependency versions and model evaluation results after running on the chosen dataset.
3. Add a reproducible experiment log and compare KNN with a simple baseline.
4. Add proper validation and calibration only if there is a clearly defined use case and sufficient data.

## License

Add a license before inviting reuse. Also confirm the dataset's own terms separately; a code license does not automatically license the dataset.

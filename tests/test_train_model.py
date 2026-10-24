import tempfile
import unittest
from pathlib import Path

import pandas as pd

from train_model import FEATURES, ZERO_AS_MISSING, load_dataset


class LoadDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "diabetes.csv"
        rows = []
        for outcome in (0, 1):
            rows.append({
                "Pregnancies": 0,
                "Glucose": 0,
                "BloodPressure": 70,
                "SkinThickness": 20,
                "Insulin": 0,
                "BMI": 25.0,
                "DiabetesPedigreeFunction": 0.4,
                "Age": 30,
                "Outcome": outcome,
            })
        pd.DataFrame(rows).to_csv(self.path, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_zero_placeholders_become_missing_but_zero_pregnancies_remains(self):
        X, y = load_dataset(self.path)
        self.assertTrue(pd.isna(X.loc[0, "Glucose"]))
        self.assertTrue(pd.isna(X.loc[0, "Insulin"]))
        self.assertEqual(X.loc[0, "Pregnancies"], 0)
        self.assertEqual(list(X.columns), FEATURES)
        self.assertEqual(set(y.tolist()), {0, 1})

    def test_target_must_be_binary(self):
        data = pd.read_csv(self.path)
        data.loc[0, "Outcome"] = 2
        data.to_csv(self.path, index=False)
        with self.assertRaises(ValueError):
            load_dataset(self.path)

    def test_required_columns_are_checked(self):
        pd.DataFrame({"Outcome": [0, 1]}).to_csv(self.path, index=False)
        with self.assertRaises(ValueError):
            load_dataset(self.path)


if __name__ == "__main__":
    unittest.main()

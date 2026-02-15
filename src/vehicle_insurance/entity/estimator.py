from typing import List, Tuple
from pandas import DataFrame
import logging
from imblearn.pipeline import Pipeline as ImbPipeline
from vehicle_insurance.constants import MODEL_CONFIG_FILE
from vehicle_insurance.utils.main_utils import read_yaml_file
from vehicle_insurance.exception import MyException
import sys
import numpy as np
logger = logging.getLogger("Estimator")


class MyModel:
    def __init__(
        self,
        pipeline: ImbPipeline,
        expected_features: List[str],
        schema_cols: List[str]
    ) -> None:
        self.pipeline: ImbPipeline = pipeline
        self._model_schema: dict = read_yaml_file(MODEL_CONFIG_FILE)
        self._validate_pipeline()
        self.expected_features: List[str] = expected_features
        self.schema_cols: List[str] = schema_cols

    def _get_final_estimator(self) -> object:
        """Return final estimator if it's a Pipeline; otherwise return the object itself."""
        if hasattr(self.pipeline, "steps"):  # sklearn/imblearn Pipeline
            return self.pipeline.steps[-1][1]
        return self.pipeline

    def _validate_pipeline(self) -> None:
        # pipeline must have predict_proba (Pipeline will proxy to last step)
        if not hasattr(self.pipeline, "predict_proba"):
            last = self._get_final_estimator()
            raise TypeError(
                "MyModel expects a fitted MODEL pipeline that supports predict_proba(). "
                "You likely passed a preprocessor-only pipeline.\n"
                f"Got object type: {type(self.pipeline).__name__}\n"
                f"Final estimator type: {type(last).__name__}\n"
                "Fix: build/save a full pipeline: (cleaning -> preprocess -> model)."
            )

    def _validate_df(self, df: DataFrame) -> None:
        missing = set(self.expected_features) - set(df.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")

        extra = [c for c in df.columns if c not in self.schema_cols]
        if extra:
            raise Exception(f"data has some extra columns: {extra}")

    def _validate_threshold(self) -> float:
        thr = self._model_schema.get("threshold", None)
        if thr is None:
            raise KeyError("MODEL_CONFIG_FILE must contain a 'threshold' key.")
        if not (0.0 <= float(thr) <= 1.0):
            raise ValueError(f"threshold must be between 0 and 1. Got: {thr}")
        return float(thr)

    def predict(self, df: DataFrame) -> List[int]:
        try:
            self._validate_df(df)
            predictions = self.pipeline.predict_proba(df)[:, -1]
            thr = self._validate_threshold()
            preds_tuned = (predictions >= thr).astype(int).tolist()
            return preds_tuned
        except Exception as e:
            raise MyException(e, sys) from e

    def predict_proba(self, df: DataFrame) -> List[float]:
        try:
            self._validate_df(df)
            proba = self.pipeline.predict_proba(df)[:, -1].tolist()
            return proba
        except Exception as e:
            raise MyException(e, sys) from e

    def predict_with_confidence_score(self, df: DataFrame) -> Tuple[List[float], List[int]]:
        try:
            self._validate_df(df)
            proba = self.pipeline.predict_proba(df)
            p_yes = proba[:, 1]
            p_no = proba[:, 0]

            thr = self._validate_threshold()
            preds_tuned = (p_yes >= thr).astype(int)  # one prediction per row

            # confidence per row: probability of the predicted class
            confidence = np.where(preds_tuned == 1, p_yes, p_no)

            return confidence.tolist(), preds_tuned.tolist()
        except Exception as e:
            raise MyException(e, sys) from e



    def __repr__(self) -> str:
        return f"{type(self.pipeline).__name__}()"

    def __str__(self) -> str:
        return f"{type(self.pipeline).__name__}()"


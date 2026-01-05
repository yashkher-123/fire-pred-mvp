#started 1/4/2026

import pandas as pd
import joblib
import dill
import numpy as np

class MVPService:
    """
    Pure prediction + explanation service.
    - Loads artifacts once
    - Accepts feature dictionaries
    - Returns JSON-serializable outputs
    """

    def __init__(self):
        # Load artifacts ONCE (server startup)
        self.scalers = joblib.load("scalers.pkl")
        self.xgb_model = joblib.load("xgb_model.pkl")

        with open("lime_explainer.dill", "rb") as f:
            bundle = dill.load(f)

        self.explainer = bundle["explainer"]
        self.predict_fn = bundle["predict_fn"]

        # Setup scalers
        self.std_scaler = self.scalers["standard_scaler"]
        self.pwr_scaler = self.scalers["power_scaler"]

        self.standard_cols = [
            "temp_max_F",
            "humidity_pct",
            "windspeed_mph",
            "ndvi",
            "slope"
        ]

        self.power_cols = [
            "precip_in",
            "pop_density"
        ]

        # Canonical feature order (CRITICAL for model correctness)
        self.feature_order = self.xgb_model.get_booster().feature_names


    def _prepare_input(self, features: dict):
        """
        Convert feature dict → scaled dataframe → numpy instance
        """
        df = pd.DataFrame([features])[self.feature_order]
        df_nonscaled = df.copy()

        # Apply scalers
        df[self.standard_cols] = self.std_scaler.transform(df[self.standard_cols])
        df[self.power_cols] = self.pwr_scaler.transform(df[self.power_cols])

        x_instance = df.iloc[0].to_numpy()

        return df, df_nonscaled, x_instance

    def predict(self, features: dict):
        """
        Return model prediction (log-scale + acres)
        """
        df, _, _ = self._prepare_input(features)

        pred_log = float(self.xgb_model.predict(df)[0])
        pred_acres = float(10 ** pred_log)

        return {
            "prediction_log": pred_log,
            "prediction_acres": pred_acres
        }

    def explain(self, features: dict, top_k=10):
        """
        Return LIME explanation data (no plots)
        """
        df, df_nonscaled, x_instance = self._prepare_input(features)

        explanation = self.explainer.explain_instance(
            x_instance,
            self.predict_fn
        )

        # Extract explanation as structured data
        lime_data = []
        for feature, weight in explanation.as_list()[:top_k]:
            lime_data.append({
                "feature": feature,
                "weight": float(weight)
            })

        pred_log = float(self.xgb_model.predict(df)[0])
        pred_acres = float(10 ** pred_log)

        return {
            "prediction_log": pred_log,
            "prediction_acres": pred_acres,
            "lime_explanation": lime_data,
            "input_features": df_nonscaled.iloc[0].to_dict()
        }
    

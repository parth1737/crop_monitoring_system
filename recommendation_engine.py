"""
recommendation_engine.py
========================
Pesticide & Spraying Recommendation Engine for Soybean Health Management.
Matches detected YOLO disease classes with pesticide database guidelines.
"""

import os
import pandas as pd

class PesticideRecommendationEngine:
    def __init__(self, csv_path=None):
        if csv_path is None:
            csv_path = r"c:\Users\ASUS\Downloads\ADRI\crop_monitoring_system\pesticide_knowledge_base.csv"
        self.csv_path = csv_path
        self.kb_df = self._load_kb()

    def _load_kb(self):
        if os.path.exists(self.csv_path):
            try:
                return pd.read_csv(self.csv_path)
            except Exception as e:
                print(f"⚠️ Error reading KB CSV: {e}")
                return pd.DataFrame()
        else:
            print(f"⚠️ Knowledge base CSV not found at {self.csv_path}")
            return pd.DataFrame()

    def get_recommendation(self, disease_class):
        """
        Retrieves tailored recommendation dict for a detected disease class string.
        """
        if self.kb_df.empty:
            return {"error": "Knowledge Base unavailable"}

        # Match exact or partial class name
        match = self.kb_df[self.kb_df['disease_class'].str.lower() == str(disease_class).lower()]
        if match.empty:
            # Fuzzy match
            match = self.kb_df[self.kb_df['disease_class'].apply(lambda x: x.lower() in str(disease_class).lower() or str(disease_class).lower() in x.lower())]

        if not match.empty:
            row = match.iloc[0]
            return {
                "disease_name": row.get("disease_name", disease_class),
                "recommended_pesticide": row.get("recommended_pesticide", "Consult Agronomist"),
                "active_ingredient": row.get("active_ingredient", "N/A"),
                "dosage_per_ha": row.get("dosage_per_hectare", "N/A"),
                "water_volume_per_ha": row.get("water_volume_per_ha", "N/A"),
                "spray_interval_days": row.get("spray_interval_days", "N/A"),
                "safety_buffer_m": row.get("safety_buffer_meters", "N/A"),
                "application_notes": row.get("application_notes", "Follow standard GAP guidelines.")
            }
        else:
            return {
                "disease_name": disease_class,
                "recommended_pesticide": "Broad-Spectrum Fungicide (Azoxystrobin/Pyraclostrobin)",
                "active_ingredient": "Azoxystrobin 23% SC",
                "dosage_per_ha": "300-400 ml",
                "water_volume_per_ha": "500 L",
                "spray_interval_days": "14",
                "safety_buffer_m": "10",
                "application_notes": "Generic broad-spectrum protective spray."
            }

# core/identifier.py
"""Shared, stable ID generation used by both DataCleaningService and SpecTextBuilder,
so phone_cleaned.csv and phones_grouped_for_rag.csv can always be joined on model_id."""
import hashlib

def make_model_id(brand: str, model: str) -> str:
    key = f"{str(brand).strip().lower()}|{str(model).strip().lower()}"
    return hashlib.md5(key.encode()).hexdigest()[:12]
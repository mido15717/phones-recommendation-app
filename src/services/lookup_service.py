# services/listing_service.py
from __future__ import annotations
import pandas as pd


class ListingService:
    """Looks up real store listings (price, stock, URL) for a phone model."""

    def __init__(self, cleaned_df: pd.DataFrame) -> None:
        self.df = cleaned_df

    @staticmethod
    def _clean(value):
        """Convert pandas NaN/NaT to None so it's JSON-serializable."""
        if pd.isna(value):
            return None
        return value

    def get_listings(self, brand: str, model: str, limit: int = 10) -> list[dict]:
        matches = self.df[
            (self.df["brand"].str.lower() == str(brand).lower())
            & (self.df["model"].str.lower() == str(model).lower())
        ]
        if matches.empty:
            matches = self.df[self.df["model"].str.lower() == str(model).lower()]

        matches = matches.sort_values("price_egp").head(limit)
        return [
            {
                "source_site": self._clean(row["source_site"]),
                "seller_name": self._clean(row["seller_name"]),
                "product_url": self._clean(row["product_url"]),
                "price_egp": self._clean(row["price_egp"]),
                "original_price_egp": self._clean(row["original_price_egp"]),
                "storage_gb": self._clean(row["storage_gb"]),
                "ram_gb": self._clean(row["ram_gb"]),
                "color": self._clean(row["color"]),
                "network": self._clean(row["network"]),
                "sim_config": self._clean(row["sim_config"]),
                "in_stock": bool(row["in_stock"]) if pd.notna(row["in_stock"]) else False,
            }
            for _, row in matches.iterrows()
        ]
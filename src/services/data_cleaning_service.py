import pandas as pd


class DataCleaningService:
    """Cleans raw scraped phone listings into a structured DataFrame."""

    NETWORK_TOKENS = {"5G", "4G", "4G LTE", "3G", "LTE"}

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def parse_model_name(self, raw: str) -> dict:
        result = {"model": None, "storage_gb": None, "ram_gb": None,
                  "network": None, "sim_config": None, "color": None}

        if " - " in raw:
            main_part, color = raw.rsplit(" - ", 1)
            result["color"] = color.strip()
        else:
            main_part = raw

        parts = [p.strip() for p in main_part.split(",")]
        result["model"] = parts[0]

        for p in parts[1:]:
            if p.endswith("GB"):
                number_str = p.replace("GB", "").strip()
                if number_str.isdigit():
                    if result["storage_gb"] is None:
                        result["storage_gb"] = int(number_str)
                    else:
                        result["ram_gb"] = int(number_str)
            elif p.endswith("Terabyte"):
                number_str = p.replace("Terabyte", "").strip()
                if number_str.isdigit():
                    result["storage_gb"] = int(number_str) * 1024
            elif "SIM" in p:
                result["sim_config"] = p
            elif p in self.NETWORK_TOKENS:
                result["network"] = p

        return result

    def parse_all(self, df: pd.DataFrame) -> pd.DataFrame:
        parsed = df["model_name"].apply(self.parse_model_name).apply(pd.Series)
        out = pd.concat([df, parsed], axis=1)
        out["is_feature_phone"] = out["storage_gb"].isna() & out["ram_gb"].isna()
        return out

    def fix_ram_storage_swap(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        outliers = df[df["ram_gb"] > df["storage_gb"]]
        self._log(f"Outliers (RAM > storage): {len(outliers)}")

        for _, row in outliers.iterrows():
            self._log(
                f"Model: {row['variant_id']}, {row['model']}, "
                f"RAM: {row['ram_gb']}GB, Storage: {row['storage_gb']}GB"
            )

        outlier_ids = outliers["variant_id"].tolist()
        mask = df["variant_id"].isin(outlier_ids)
        df.loc[mask, ["ram_gb", "storage_gb"]] = df.loc[mask, ["storage_gb", "ram_gb"]].values

        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full pipeline: parse model names, fix RAM/storage swaps, drop scrape metadata."""
        df = self.parse_all(df)
        df = self.fix_ram_storage_swap(df)
        df = df.drop(columns=["model_name", "scraped_at", "is_feature_phone"], errors="ignore")
        return df
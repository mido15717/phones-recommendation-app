"""
Builds phones_grouped_for_rag.csv from phone_with_specs.csv.

Pipeline:
  1. Load raw variant-level specs data
  2. Parse freeform key_features_text into normalized kf_* fields
  3. Merge parsed fields back onto the dataframe
  4. Build one spec_text string per row (for embedding)
  5. Generate a stable model_id (join key back to phone_cleaned.csv)
  6. Group by model, aggregate to one row per model
  7. Save
"""

import hashlib
import pandas as pd
import re


class specsbuilder:
    LABEL_MAP = {
        'brand': ['brand'],
        'model_name': ['model name', 'model', 'model number'],
        'display': ['display', 'resolution'],
        'processor': ['processor', 'chipset'],
        'rear_camera': ['rear camera', 'main camera', 'rare camera', 'camera', 'primary camera'],
        'front_camera': ['front camera', 'front camera for stunning selfies'],
        'battery': ['battery capacity', 'battery', 'charging power', 'fast charging that lasts all day'],
        'num_sim_cards': ['number of sim cards', 'number of sim', 'number of simcards', 'number of sim card'],
        'sim_type': ['sim type', 'type'],
        'connectivity': ['connectivity', 'faster connectivity', 'unlimited speed and connectivity', 'network'],
        'operating_system': ['operating system'],
        'other_features': ['other features', 'sensors', 'features', 'key features'],
        'memory': ['memory', 'ram', 'storage capacity'],
        'video': ['video', 'video recording', 'shooting modes'],
        'dimensions': ['dimensions'],
        'color': ['color'],
    }

    EXCLUDE_COLS = {
        "source_site", "product_url", "image_url", "variant_id", "product_id",
        "price_egp", "original_price_egp", "in_stock", "is_best_seller", "is_new_arrival",
        "seller_name", "category_l3", "model", "storage_gb", "ram_gb", "network",
        "sim_config", "color", "brand",
    }

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.raw_to_canon = {}
        for canon, variants in self.LABEL_MAP.items():
            for v in variants:
                self.raw_to_canon[v.lower()] = canon

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # ---------- Step 2: parse freeform key_features_text ----------
    def parse_key_features(self, text: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            return {}

        parsed = {}
        current_label = None
        buffer = []

        def flush():
            if current_label and buffer:
                content = " ".join(b.strip("- ").strip() for b in buffer if b.strip())
                if current_label in parsed:
                    parsed[current_label] += " " + content
                else:
                    parsed[current_label] = content

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^([A-Za-z][A-Za-z ]{2,40}):\s*(.*)", line)
            if m:
                raw_label = m.group(1).strip().lower()
                rest = m.group(2).strip()
                canon = self.raw_to_canon.get(raw_label)
                if canon:
                    flush()
                    current_label = canon
                    buffer = [rest] if rest else []
                    continue
                else:
                    if rest:
                        buffer.append(rest)
                    continue
            else:
                buffer.append(line)

        flush()
        return parsed

    # ---------- Step 4: build spec_text per row ----------
    @staticmethod
    def _row_to_spec_text(row: pd.Series, text_cols: list[str]) -> str:
        parts = []
        for c in text_cols:
            val = row[c]
            if pd.notna(val) and str(val).strip():
                parts.append(f"{c}: {val}")
        return " | ".join(parts)

    # ---------- Step 5: stable model_id ----------
    @staticmethod
    def make_model_id(brand: str, model: str) -> str:
        key = f"{str(brand).strip().lower()}|{str(model).strip().lower()}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    # ---------- full pipeline ----------
    def build_grouped(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Step 2 + 3: parse key_features_text, merge parsed kf_* fields back
        parsed_series = df["key_features_text"].apply(self.parse_key_features)
        parsed_df = pd.json_normalize(parsed_series)
        parsed_df.columns = [f"kf_{c}" for c in parsed_df.columns]

        df = pd.concat([df.reset_index(drop=True), parsed_df.reset_index(drop=True)], axis=1)
        df = df.drop(columns=["key_features_text"])

        # Step 4: build spec_text
        text_cols = [c for c in df.columns if c not in self.EXCLUDE_COLS]
        self._log(f"spec_text built from {len(text_cols)} columns")
        df["spec_text"] = df.apply(lambda row: self._row_to_spec_text(row, text_cols), axis=1)

        # Step 5: stable model_id (join key to phone_cleaned.csv variants)
        df["model_id"] = df.apply(lambda r: self.make_model_id(r["brand"], r["model"]), axis=1)

        # Step 6: group by model, aggregate to one row per model
        grouped = df.groupby(["model_id", "brand", "model"]).agg(
            price_min=("price_egp", "min"),
            price_max=("price_egp", "max"),
            storage_options=("storage_gb", lambda x: sorted(x.dropna().unique().tolist())),
            ram_options=("ram_gb", lambda x: sorted(x.dropna().unique().tolist())),
            network=("network", lambda x: sorted(x.dropna().unique().tolist())),
            category=("category_l3", "first"),
            in_stock=("in_stock", "max"),
            spec_text=("spec_text", lambda texts: max([t for t in texts if t], key=len, default="")),
        ).reset_index()

        # validation: catch both NaN and empty-string spec_text (the == 0 check misses NaN)
        empty_mask = grouped["spec_text"].isna() | (grouped["spec_text"].str.len() == 0)
        self._log(f"Done. {grouped.shape[0]} unique models, {grouped.shape[1]} columns.")
        self._log(f"Empty spec_text rows: {empty_mask.sum()}")
        if empty_mask.sum() > 0:
            self._log("Models with empty spec_text (need investigation):")
            self._log(grouped.loc[empty_mask, ["brand", "model"]].to_string())

        return grouped


if __name__ == "__main__":
    df = pd.read_csv("phone_with_specs.csv")

    builder = SpecTextBuilder(verbose=True)
    grouped = builder.build_grouped(df)

    grouped.to_csv("phones_grouped_for_rag.csv", index=False)
    print(f"\nSaved phones_grouped_for_rag.csv ({len(grouped)} rows)")
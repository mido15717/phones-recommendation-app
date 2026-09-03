"""
Builds phones_grouped_for_rag.csv from phone_with_specs.csv.

Pipeline:
  1. Load raw variant-level specs data
  2. Parse freeform key_features_text into normalized kf_* fields
  3. Merge parsed fields back onto the dataframe
  4. Generate a stable model_id
  5. Group variants by model
  6. Aggregate ALL semantic kf_* fields across variants
  7. Build one clean model-level spec_text
  8. Validate the generated spec_text
  9. Save

Important:
  - Structured fields such as price, RAM, storage, network, etc.
    remain separate from spec_text.
  - spec_text contains semantic/product information intended
    for embedding and semantic retrieval.
"""

import pandas as pd
import re
from core.logger import logger
from core.identifiers import make_model_id

class SpecTextBuilder:

    # ---------------------------------------------------------
    # Canonical labels
    # ---------------------------------------------------------

    LABEL_MAP = {
        "brand": ["brand"],

        "model_name": [
            "model name",
            "model",
            "model number",
        ],

        "display": [
            "display",
            "resolution",
        ],

        "processor": [
            "processor",
            "chipset",
        ],

        "rear_camera": [
            "rear camera",
            "main camera",
            "rare camera",
            "camera",
            "primary camera",
        ],

        "front_camera": [
            "front camera",
            "front camera for stunning selfies",
        ],

        "battery": [
            "battery capacity",
            "battery",
            "charging power",
            "fast charging that lasts all day",
        ],

        "num_sim_cards": [
            "number of sim cards",
            "number of sim",
            "number of simcards",
            "number of sim card",
        ],

        "sim_type": [
            "sim type",
            # Removed generic "type"
        ],

        "connectivity": [
            "connectivity",
            "faster connectivity",
            "unlimited speed and connectivity",
        ],

        "operating_system": [
            "operating system",
        ],

        "other_features": [
            "other features",
            "sensors",
            "features",
            "key features",
        ],

        "memory": [
            "memory",
            "ram",
            "storage capacity",
        ],

        "video": [
            "video",
            "video recording",
            "shooting modes",
        ],

        "dimensions": [
            "dimensions",
        ],

        "color": [
            "color",
        ],
    }

    # ---------------------------------------------------------
    # Columns that should NOT become semantic spec_text
    # ---------------------------------------------------------

    EXCLUDE_COLS = {
        "source_site",
        "product_url",
        "image_url",
        "variant_id",
        "product_id",

        # Pricing
        "price_egp",
        "original_price_egp",

        # Inventory / business information
        "in_stock",
        "is_best_seller",
        "is_new_arrival",
        "seller_name",

        # Category
        "category_l3",

        # Structured phone attributes
        "model",
        "storage_gb",
        "ram_gb",
        "network",
        "sim_config",
        "color",
        "brand",

        # Internal/generated columns
        "model_id",
        "spec_text",
    }

    # ---------------------------------------------------------
    # Human-readable labels for final spec_text
    # ---------------------------------------------------------

    SPEC_LABELS = {
        "brand": "Brand",
        "model_name": "Model",
        "display": "Display",
        "processor": "Processor",
        "rear_camera": "Rear Camera",
        "front_camera": "Front Camera",
        "battery": "Battery",
        "num_sim_cards": "SIM Cards",
        "sim_type": "SIM Type",
        "connectivity": "Connectivity",
        "operating_system": "Operating System",
        "other_features": "Other Features",
        "memory": "Memory",
        "video": "Video",
        "dimensions": "Dimensions",
        "color": "Color",
    }

    def __init__(self):

        # Example:
        # "Battery Capacity" -> "battery"
        # "Chipset" -> "processor"

        self.raw_to_canon = {}

        for canon, variants in self.LABEL_MAP.items():
            for variant in variants:
                self.raw_to_canon[self._normalize_label(variant)] = canon

    # =========================================================
    # NORMALIZATION HELPERS
    # =========================================================

    @staticmethod
    def _normalize_label(label: str) -> str:
        """
        Normalize a raw label so different formatting styles
        map to the same canonical label.

        Examples:
            "Battery Capacity" -> "battery capacity"
            "battery-capacity" -> "battery capacity"
            " Battery   Capacity " -> "battery capacity"
        """

        if not isinstance(label, str):
            return ""

        label = label.strip().lower()

        # Replace separators with spaces
        label = re.sub(r"[-_/]+", " ", label)

        # Remove excessive whitespace
        label = re.sub(r"\s+", " ", label)

        return label.strip()

    @staticmethod
    def _normalize_value(value: str) -> str:
        """
        Normalize a specification value.

        This does not aggressively modify the actual specification.
        It mainly removes formatting noise.
        """

        if not isinstance(value, str):
            return ""

        value = value.strip()

        # Remove bullet characters
        value = re.sub(r"^[•●▪◦\-]+\s*", "", value)

        # Collapse whitespace
        value = re.sub(r"\s+", " ", value)

        return value.strip()

    @staticmethod
    def _deduplicate_values(values: list[str]) -> list[str]:
        """
        Deduplicate values while preserving their original order.
        Comparison is case-insensitive.
        """

        result = []
        seen = set()

        for value in values:

            value = SpecTextBuilder._normalize_value(value)

            if not value:
                continue

            key = value.casefold()

            if key not in seen:
                seen.add(key)
                result.append(value)

        return result

    # =========================================================
    # STEP 2: PARSE key_features_text
    # =========================================================

    def parse_key_features(self, text: str) -> dict:
        """
        Parse free-form key_features_text into canonical fields.

        Example input:

            Display: 6.7 inch AMOLED
            Processor: Snapdragon 8 Gen 3
            Battery Capacity: 5000mAh
            Rear Camera: 50MP
            Unknown Feature: Something

        Known labels are mapped to canonical names.

        Unknown labels are preserved instead of silently discarded.
        """

        if not isinstance(text, str) or not text.strip():
            return {}

        parsed = {}

        current_label = None
        buffer = []

        def flush():

            nonlocal current_label, buffer

            if current_label is None:
                buffer = []
                return

            values = []

            for item in buffer:

                item = self._normalize_value(item)

                if item:
                    values.append(item)

            if not values:
                buffer = []
                return

            # Deduplicate values within the current field
            values = self._deduplicate_values(values)

            content = "; ".join(values)

            if current_label in parsed:

                existing = parsed[current_label]

                combined = existing.split("; ") + content.split("; ")

                parsed[current_label] = "; ".join(
                    self._deduplicate_values(combined)
                )

            else:
                parsed[current_label] = content

            buffer = []

        # -----------------------------------------------------
        # Parse line by line
        # -----------------------------------------------------

        for raw_line in text.splitlines():

            line = raw_line.strip()

            if not line:
                continue

            # Much more permissive than the old regex.
            #
            # Old:
            # ^([A-Za-z][A-Za-z ]{2,40}):\s*(.*)
            #
            # New:
            # everything before ":" is considered a possible label.
            match = re.match(r"^([^:]{1,80}):\s*(.*)$", line)

            if match:

                raw_label = self._normalize_label(match.group(1))
                rest = self._normalize_value(match.group(2))

                canon = self.raw_to_canon.get(raw_label)

                if canon:

                    # Finish previous field
                    flush()

                    current_label = canon
                    buffer = []

                    if rest:
                        buffer.append(rest)

                else:

                    # Unknown label:
                    #
                    # Do NOT silently throw away the label.
                    #
                    # Preserve it using a normalized dynamic key.
                    flush()

                    unknown_label = re.sub(
                        r"[^a-z0-9]+",
                        "_",
                        raw_label
                    ).strip("_")

                    if unknown_label:

                        current_label = f"unknown_{unknown_label}"
                        buffer = []

                        if rest:
                            buffer.append(rest)

                continue

            # Continuation line
            if current_label is not None:
                buffer.append(line)

        # Flush final field
        flush()

        return parsed

    # =========================================================
    # STEP 3: MERGE PARSED FIELDS
    # =========================================================

    def _parse_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:

        if "key_features_text" not in df.columns:
            raise ValueError(
                "Input dataframe must contain 'key_features_text'."
            )

        parsed_series = df["key_features_text"].apply(
            self.parse_key_features
        )

        parsed_df = pd.json_normalize(parsed_series)

        if not parsed_df.empty:
            parsed_df.columns = [
                f"kf_{column}"
                for column in parsed_df.columns
            ]

        result = pd.concat(
            [
                df.reset_index(drop=True),
                parsed_df.reset_index(drop=True),
            ],
            axis=1,
        )

        result = result.drop(
            columns=["key_features_text"],
            errors="ignore"
        )

        return result

    # =========================================================
    # STEP 5: MODEL ID
    # =========================================================

    @staticmethod
    def _add_model_id(df: pd.DataFrame) -> pd.DataFrame:

        required = {"brand", "model"}

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing required columns for model_id: {missing}"
            )

        df = df.copy()

        df["model_id"] = df.apply(
            lambda row: make_model_id(
                row["brand"],
                row["model"]
            ),
            axis=1,
        )

        return df

    # =========================================================
    # STEP 6: MERGE ALL SPECIFICATIONS
    # =========================================================

    @staticmethod
    def _merge_spec_values(series: pd.Series) -> str:
        """
        Merge all specification values from all variants.

        Example:

            Variant 1:
                Snapdragon 8 Gen 3

            Variant 2:
                Snapdragon 8 Gen 3

            Variant 3:
                Snapdragon 8 Gen 3

        Result:

            Snapdragon 8 Gen 3

        Duplicate values are removed.
        """

        values = []

        for value in series.dropna():

            value = str(value).strip()

            if not value:
                continue

            # Values inside parsed fields may already be separated
            # by semicolons.
            parts = value.split(";")

            for part in parts:

                part = SpecTextBuilder._normalize_value(part)

                if part:
                    values.append(part)

        values = SpecTextBuilder._deduplicate_values(values)

        return "; ".join(values)

    # =========================================================
    # STEP 7: BUILD FINAL MODEL-LEVEL spec_text
    # =========================================================

    def _build_model_spec_text(
        self,
        row: pd.Series,
        kf_columns: list[str],
    ) -> str:
        """
        Build one clean semantic description for a model.

        This happens AFTER grouping, meaning information from
        every variant can contribute to the final text.
        """

        parts = []

        for column in kf_columns:

            value = row.get(column)

            if pd.isna(value):
                continue

            value = str(value).strip()

            if not value:
                continue

            canonical_name = column.replace("kf_", "", 1)

            # Known canonical field
            label = self.SPEC_LABELS.get(
                canonical_name,
                canonical_name.replace("_", " ").title()
            )

            parts.append(
                f"{label}: {value}"
            )

        return " | ".join(parts)

    # =========================================================
    # STEP 8: VALIDATION
    # =========================================================

    def _validate_spec_text(
        self,
        grouped: pd.DataFrame,
    ) -> None:
        """
        Validate generated model-level spec_text.
        """

        if "spec_text" not in grouped.columns:
            logger.warning("spec_text column was not generated.")
            return

        empty_mask = (
            grouped["spec_text"].isna()
            | grouped["spec_text"].astype(str).str.strip().eq("")
        )

        empty_count = int(empty_mask.sum())

        if empty_count > 0:

            logger.warning(
                f"{empty_count} models have empty spec_text."
            )

            logger.warning(
                "Affected models:\n"
                + grouped.loc[
                    empty_mask,
                    ["brand", "model"]
                ].to_string(index=False)
            )

        else:

            logger.info(
                "Validation passed: no empty spec_text rows."
            )

        # Check for duplicate-looking separators
        malformed_separator = grouped[
            grouped["spec_text"]
            .astype(str)
            .str.contains(r"\|\s*\|", regex=True)
        ]

        if not malformed_separator.empty:

            logger.warning(
                f"{len(malformed_separator)} models contain "
                "malformed separators in spec_text."
            )

        # Basic statistics
        lengths = grouped["spec_text"].astype(str).str.len()

        logger.info(
            "spec_text length statistics: "
            f"min={lengths.min()}, "
            f"avg={lengths.mean():.1f}, "
            f"max={lengths.max()}"
        )

    # =========================================================
    # FULL PIPELINE
    # =========================================================

    def build_grouped(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        logger.info(
            f"Building grouped RAG dataset "
            f"from {len(df)} raw spec rows"
        )

        # -----------------------------------------------------
        # STEP 2 + 3
        # Parse key_features_text
        # -----------------------------------------------------

        df = self._parse_dataframe(df)

        logger.info(
            "Parsed key_features_text into normalized kf_* fields."
        )

        # -----------------------------------------------------
        # STEP 5
        # Generate stable model_id
        # -----------------------------------------------------

        df = self._add_model_id(df)

        # -----------------------------------------------------
        # Identify all parsed semantic fields
        # -----------------------------------------------------

        kf_columns = [
            column
            for column in df.columns
            if column.startswith("kf_")
        ]

        logger.info(
            f"Found {len(kf_columns)} semantic specification fields."
        )

        # -----------------------------------------------------
        # STEP 6
        # Group variants by model
        # -----------------------------------------------------

        aggregation = {
            "price_egp": (
                "price_egp",
                "min",
            ),

            "price_max": (
                "price_egp",
                "max",
            ),

            "storage_options": (
                "storage_gb",
                lambda x: sorted(
                    x.dropna()
                    .unique()
                    .tolist()
                ),
            ),

            "ram_options": (
                "ram_gb",
                lambda x: sorted(
                    x.dropna()
                    .unique()
                    .tolist()
                ),
            ),

            "network": (
                "network",
                lambda x: sorted(
                    x.dropna()
                    .unique()
                    .tolist()
                ),
            ),

            "category": (
                "category_l3",
                "first",
            ),

            "in_stock": (
                "in_stock",
                "max",
            ),
        }

        # -----------------------------------------------------
        # CRITICAL CHANGE:
        #
        # Aggregate EVERY kf_* field across ALL variants.
        #
        # We do NOT select the longest spec_text anymore.
        # -----------------------------------------------------

        for column in kf_columns:

            aggregation[column] = (
                column,
                self._merge_spec_values,
            )

        grouped = (
            df
            .groupby(
                [
                    "model_id",
                    "brand",
                    "model",
                ],
                dropna=False,
            )
            .agg(**aggregation)
            .reset_index()
        )

        # Rename price columns
        grouped = grouped.rename(
            columns={
                "price_egp": "price_min",
            }
        )

        # -----------------------------------------------------
        # STEP 7
        # Build final model-level spec_text
        # -----------------------------------------------------

        grouped["spec_text"] = grouped.apply(
            lambda row: self._build_model_spec_text(
                row,
                kf_columns,
            ),
            axis=1,
        )

        # -----------------------------------------------------
        # Remove temporary kf_* columns from final output
        #
        # Their information is now represented inside
        # spec_text.
        # -----------------------------------------------------

        grouped = grouped.drop(
            columns=kf_columns,
            errors="ignore",
        )

        # -----------------------------------------------------
        # Reorder columns for readability
        # -----------------------------------------------------

        preferred_order = [
            "model_id",
            "brand",
            "model",
            "price_min",
            "price_max",
            "storage_options",
            "ram_options",
            "network",
            "category",
            "in_stock",
            "spec_text",
        ]

        remaining = [
            column
            for column in grouped.columns
            if column not in preferred_order
        ]

        grouped = grouped[
            [
                column
                for column in preferred_order
                if column in grouped.columns
            ]
            + remaining
        ]

        # -----------------------------------------------------
        # STEP 8
        # Validate
        # -----------------------------------------------------

        self._validate_spec_text(grouped)

        logger.info(
            f"Done. "
            f"{grouped.shape[0]} unique models, "
            f"{grouped.shape[1]} columns."
        )

        return grouped


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent  # -> src/

    input_path = BASE_DIR / "data" / "raw_data" / "phone_with_specs.csv"

    output_path = (
        BASE_DIR / "data" / "processed_data"
        / "phones_grouped_for_rag_v1.csv"
    )

    logger.info(f"Loading {input_path}")

    df = pd.read_csv(input_path)

    builder = SpecTextBuilder()

    grouped = builder.build_grouped(df)

    grouped.to_csv(
        output_path,
        index=False,
    )

    logger.info(
        f"Saved {output_path} "
        f"({len(grouped)} rows)"
    )
import pandas as pd
from core.logger import logger
from services.data_cleaning_service import DataCleaningService
from services.specsbuilder import SpecTextBuilder


class Validate_Data:
    CLEANED_SCHEMA = {
        "source_site": "str", "product_url": "str", "image_url": "str",
        "brand": "str", "variant_id": "str", "product_id": "str",
        "price_egp": "float", "original_price_egp": "float", "in_stock": "bool",
        "is_best_seller": "bool", "is_new_arrival": "bool", "seller_name": "str",
        "category_l3": "str", "model": "str", "storage_gb": "float",
        "ram_gb": "float", "network": "str", "sim_config": "str", "color": "str",
        "model_id": "str",
    }
    CLEANED_DEDUPE_COL = "variant_id"
    CLEANED_RAW_SIGNATURE = {"model_name"}  # columns that mark this as raw, pre-cleaning data

    GROUPED_SCHEMA = {
        "model_id": "str", "brand": "str", "model": "str",
        "price_min": "float", "price_max": "float",
        "storage_options": "str", "ram_options": "str", "network": "str",
        "category": "str", "in_stock": "bool", "spec_text": "str",
    }
    GROUPED_DEDUPE_COL = "model_id"
    GROUPED_RAW_SIGNATURE = {"key_features_text"}  # marks this as raw specs data

    def __init__(self, cleaning_service: DataCleaningService = None, spec_builder: SpecTextBuilder = None):
        self.cleaning_service = cleaning_service or DataCleaningService()
        self.spec_builder = spec_builder or SpecTextBuilder()

    def _schema_for(self, target: str) -> dict:
        if target == "cleaned":
            return self.CLEANED_SCHEMA
        elif target == "grouped":
            return self.GROUPED_SCHEMA
        raise ValueError(f"Unknown target '{target}'. Expected 'cleaned' or 'grouped'.")

    def _is_raw(self, df: pd.DataFrame, target: str) -> bool:
        signature = self.CLEANED_RAW_SIGNATURE if target == "cleaned" else self.GROUPED_RAW_SIGNATURE
        return signature.issubset(set(df.columns))

    def _process_if_raw(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """Runs raw uploads through the real pipeline (DataCleaningService / SpecTextBuilder)
        so they end up in the correct processed schema before validation."""
        if self._is_raw(df, target):
            if target == "cleaned":
                logger.info("Detected raw variant data (model_name present) — running DataCleaningService")
                return self.cleaning_service.clean(df)
            else:
                logger.info("Detected raw spec data (key_features_text present) — running SpecTextBuilder")
                return self.spec_builder.build_grouped(df)
        return df

    def _validate_columns(self, df: pd.DataFrame, schema: dict) -> dict:
        expected_cols = set(schema.keys())
        actual_cols = set(df.columns)
        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols

        report = {
            "valid": len(missing) == 0,
            "missing_columns": sorted(missing),
            "extra_columns": sorted(extra),
        }
        if missing:
            logger.warning(f"Upload missing required columns after processing: {report['missing_columns']}")
        if extra:
            logger.info(f"Upload has extra columns not in schema (will be dropped): {report['extra_columns']}")
        return report

    def _coerce_dtypes(self, df: pd.DataFrame, schema: dict) -> tuple[pd.DataFrame, list[str]]:
        df = df.copy()
        failed_cols = []
        for col, dtype in schema.items():
            if col not in df.columns:
                continue
            try:
                if dtype == "float":
                    df[col] = pd.to_numeric(df[col], errors="raise")
                elif dtype == "bool":
                    if df[col].dtype != bool:
                        df[col] = df[col].astype(str).str.lower().map(
                            {"true": True, "false": False, "1": True, "0": False}
                        )
                        if df[col].isna().any():
                            raise ValueError("unrecognized boolean values")
                elif dtype == "str":
                    df[col] = df[col].astype(str)
            except (ValueError, TypeError) as e:
                logger.error(f"Column '{col}' failed dtype coercion to {dtype}: {e}")
                failed_cols.append(col)
        return df, failed_cols

    def validate_and_merge(self, new_df: pd.DataFrame, existing_df: pd.DataFrame, target: str) -> dict:
        """
        Ingests new_df into the given target ('cleaned' or 'grouped'):
        - if raw, runs it through the real cleaning/spec-building pipeline first
        - validates the resulting schema
        - coerces dtypes
        - merges into existing_df, deduplicating by the target's stable id column
        """
        schema = self._schema_for(target)
        dedupe_col = self.CLEANED_DEDUPE_COL if target == "cleaned" else self.GROUPED_DEDUPE_COL

        processed_df = self._process_if_raw(new_df, target)

        report = self._validate_columns(processed_df, schema)
        if not report["valid"]:
            logger.error(f"Validation failed for target='{target}'. Merge aborted.")
            return {"success": False, "merged_df": None, "report": report}

        coerced_df, failed_cols = self._coerce_dtypes(processed_df, schema)
        report["dtype_failures"] = failed_cols
        if failed_cols:
            logger.error(f"Dtype coercion failed for columns {failed_cols}. Merge aborted.")
            report["valid"] = False
            return {"success": False, "merged_df": None, "report": report}

        coerced_df = coerced_df[list(schema.keys())]

        before = len(existing_df)
        merged = pd.concat([existing_df, coerced_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=[dedupe_col], keep="last")
        after = len(merged)
        added = after - before

        logger.info(
            f"Merged into '{target}': {len(coerced_df)} incoming rows, "
            f"{added} new rows added ({len(coerced_df) - added} were duplicates/updates by '{dedupe_col}')"
        )

        return {"success": True, "merged_df": merged, "report": report}
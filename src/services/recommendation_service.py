from __future__ import annotations

import ast
import math
from typing import Any

import pandas as pd

from services.VectorStore import VectorStore
from services.Rag_Service import RAGService
from core.logger import logger


class RecommendationService:

    def __init__(
        self,
        rag_service: RAGService | None = None,
        vector_store: VectorStore | None = None,
        grouped_df: pd.DataFrame | None = None,
    ):
        self.rag = rag_service or RAGService()
        self.vector_store = vector_store or self.rag.vector_store
        self.grouped_df = grouped_df

    # ============================================================
    # Helper functions
    # ============================================================

    @staticmethod
    def _parse_list(value: Any) -> list:
        """
        Convert list-like CSV values into Python lists.

        Examples:
            "[128.0, 256.0]" -> [128.0, 256.0]
            "['4G', '5G']"    -> ['4G', '5G']
            [128, 256]       -> [128, 256]
        """

        if value is None:
            return []

        if isinstance(value, list):
            return value

        if isinstance(value, tuple):
            return list(value)

        if isinstance(value, float) and math.isnan(value):
            return []

        if isinstance(value, str):
            value = value.strip()

            if not value:
                return []

            try:
                parsed = ast.literal_eval(value)

                if isinstance(parsed, (list, tuple)):
                    return list(parsed)

                return [parsed]

            except (ValueError, SyntaxError):
                return [value]

        return [value]

    @staticmethod
    def _contains_value(
        values: Any,
        target: str,
    ) -> bool:
        """
        Case-insensitive membership check for list-like values.
        """

        parsed = RecommendationService._parse_list(values)

        target = str(target).strip().lower()

        return any(
            str(value).strip().lower() == target
            for value in parsed
        )

    @staticmethod
    def _has_numeric_minimum(
        values: Any,
        minimum: float,
    ) -> bool:
        """
        Returns True if at least one value is >= minimum.

        Example:
            [128, 256] with minimum=256 -> True
            [128] with minimum=256       -> False
        """

        parsed = RecommendationService._parse_list(values)

        for value in parsed:
            try:
                if float(value) >= minimum:
                    return True
            except (TypeError, ValueError):
                continue

        return False

    # ============================================================
    # Structured filtering
    # ============================================================

    def filter_candidates(
        self,
        df: pd.DataFrame,
        budget: float | None = None,
        brand: str | None = None,
        min_storage: float | None = None,
        min_ram: float | None = None,
        network: str | None = None,
        category: str | None = None,
        in_stock: bool | None = True,
    ) -> pd.DataFrame:
        """
        Apply deterministic/structured filters to the phone dataset.

        These filters operate on structured metadata and do NOT use
        embeddings.
        """

        if df is None or df.empty:
            return pd.DataFrame()

        candidates = df.copy()

        # --------------------------------------------------------
        # In-stock filter
        # --------------------------------------------------------

        if in_stock is not None:
            candidates = candidates[
                candidates["in_stock"] == in_stock
            ]

        # --------------------------------------------------------
        # Brand filter
        # --------------------------------------------------------

        if brand:
            brand_lower = brand.strip().lower()

            candidates = candidates[
                candidates["brand"]
                .fillna("")
                .astype(str)
                .str.lower()
                == brand_lower
            ]

        # --------------------------------------------------------
        # Category filter
        # --------------------------------------------------------

        if category:
            category_lower = category.strip().lower()

            candidates = candidates[
                candidates["category"]
                .fillna("")
                .astype(str)
                .str.lower()
                == category_lower
            ]

        # --------------------------------------------------------
        # Budget filter
        #
        # A phone is considered affordable if its minimum variant
        # price is within the user's budget.
        # --------------------------------------------------------

        if budget is not None:

            candidates = candidates.copy()

            candidates["price_min"] = pd.to_numeric(
                candidates["price_min"],
                errors="coerce",
            )

            candidates = candidates[
                candidates["price_min"] <= budget
            ]

        # --------------------------------------------------------
        # Minimum storage
        # --------------------------------------------------------

        if min_storage is not None:

            candidates = candidates[
                candidates["storage_options"].apply(
                    lambda values:
                    self._has_numeric_minimum(
                        values,
                        min_storage,
                    )
                )
            ]

        # --------------------------------------------------------
        # Minimum RAM
        # --------------------------------------------------------

        if min_ram is not None:

            candidates = candidates[
                candidates["ram_options"].apply(
                    lambda values:
                    self._has_numeric_minimum(
                        values,
                        min_ram,
                    )
                )
            ]

        # --------------------------------------------------------
        # Network
        # --------------------------------------------------------

        if network:

            candidates = candidates[
                candidates["network"].apply(
                    lambda values:
                    self._contains_value(
                        values,
                        network,
                    )
                )
            ]

        logger.info(
            f"Structured filtering returned "
            f"{len(candidates)} candidates"
        )

        return candidates

    # ============================================================
    # Hybrid retrieval (structured filter -> ANN vector search)
    # ============================================================

    def hybrid_retrieve(
        self,
        query: str,
        budget: float | None = None,
        brand: str | None = None,
        min_storage: float | None = None,
        min_ram: float | None = None,
        network: str | None = None,
        category: str | None = None,
        in_stock: bool | None = True,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Hybrid phone recommendation.

        Pipeline:

            User query
                 |
                 v
          Structured filters
                 |
                 v
          Candidate model_ids
                 |
                 v
          ANN vector search (Chroma), restricted to candidate_ids
                 |
                 v
              Top K
        """

        if not query or not query.strip():
            logger.warning("Empty recommendation query.")
            return []

        if self.grouped_df is None:
            logger.error("grouped_df has not been provided.")
            return []

        if self.vector_store is None:
            logger.error("No vector store available.")
            return []

        # --------------------------------------------------------
        # 1. Structured filtering
        # --------------------------------------------------------

        candidates = self.filter_candidates(
            df=self.grouped_df,
            budget=budget,
            brand=brand,
            min_storage=min_storage,
            min_ram=min_ram,
            network=network,
            category=category,
            in_stock=in_stock,
        )

        if candidates.empty:
            logger.info("No phones matched the structured filters.")
            return []

        candidate_ids = candidates["model_id"].astype(str).tolist()

        # --------------------------------------------------------
        # 2. Embed the query once
        # --------------------------------------------------------

        query_embedding = self.rag.embedder.encode_query(query).tolist()

        # --------------------------------------------------------
        # 3. ANN search restricted to the filtered candidates
        # --------------------------------------------------------

        chroma_results = self.vector_store.query(
            query_embedding,
            top_k=top_k,
            where={"model_id": {"$in": candidate_ids}},
        )

        ids = chroma_results.get("ids", [[]])[0]
        distances = chroma_results.get("distances", [[]])[0]

        if not ids:
            logger.info(
                "Vector store returned no matches for the filtered candidates."
            )
            return []

        # --------------------------------------------------------
        # 4. Assemble results
        # --------------------------------------------------------

        candidates_indexed = candidates.set_index(
            candidates["model_id"].astype(str)
        )

        results = []

        for model_id, distance in zip(ids, distances):

            if model_id not in candidates_indexed.index:
                continue

            row = candidates_indexed.loc[model_id]

            result = {
      "model_id": str(model_id),
      "brand": str(row["brand"]),
      "model": str(row["model"]),
      "price_min": float(row["price_min"]) if pd.notna(row["price_min"]) else None,
      "price_max": float(row["price_max"]) if pd.notna(row["price_max"]) else None,
      "storage_options": self._parse_list(row["storage_options"]),
      "ram_options": self._parse_list(row["ram_options"]),
      "network": self._parse_list(row["network"]),
      "category": str(row["category"]) if pd.notna(row["category"]) else None,
      "in_stock": bool(row["in_stock"]),
      "semantic_score": 1 - float(distance),
      "spec_text": str(row["spec_text"]) if pd.notna(row["spec_text"]) else "",
        }

            results.append(result)

        logger.info(
            f"Hybrid retrieval returned {len(results)} phones."
        )

        return results



# if __name__ == "__main__":
#     from pathlib import Path

#     BASE_DIR = Path(__file__).resolve().parent.parent  # -> src/

#     input_path = BASE_DIR / "data" / "processed_data" / "phones_grouped_for_rag_v1.csv"


#     logger.info(f"Loading {input_path}")

#     grouped_df = pd.read_csv(input_path)

#     duplicates = grouped_df[
#     grouped_df["model_id"].duplicated(keep=False)
# ].sort_values("model_id")

# print(duplicates[["model_id", "brand", "model"]])
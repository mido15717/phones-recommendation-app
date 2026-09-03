from __future__ import annotations

import ast
import math
from typing import Any

import pandas as pd

from services.Rag_Service import RAGService
from core.logger import logger


class RecommendationService:

    def __init__(
        self,
        rag_service: RAGService | None = None,
        grouped_df: pd.DataFrame | None = None,
    ):
        self.rag = rag_service or RAGService()
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
    # Hybrid retrieval
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
          Candidate phones
                 |
                 v
          Semantic similarity
                 |
                 v
             Ranking
                 |
                 v
              Top K
        """

        if not query or not query.strip():
            logger.warning("Empty recommendation query.")
            return []

        if self.grouped_df is None:
            logger.error(
                "grouped_df has not been provided."
            )
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
            logger.info(
                "No phones matched the structured filters."
            )
            return []

        # --------------------------------------------------------
        # 2. Semantic similarity
        #
        # We embed the query and compare it against spec_text.
        # --------------------------------------------------------

        query_embedding = self.rag.embedder.encode(
            [query]
        )[0]

        phone_texts = (
            candidates["spec_text"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        phone_embeddings = self.rag.embedder.encode(
            phone_texts
        )

        # --------------------------------------------------------
        # 3. Cosine similarity
        # --------------------------------------------------------

        similarities = self._cosine_similarity(
            query_embedding,
            phone_embeddings,
        )

        candidates = candidates.copy()

        candidates["semantic_score"] = similarities

        # --------------------------------------------------------
        # 4. Sort by semantic relevance
        # --------------------------------------------------------

        candidates = candidates.sort_values(
            by="semantic_score",
            ascending=False,
        )

        # --------------------------------------------------------
        # 5. Return top K
        # --------------------------------------------------------

        results = []

        for _, row in candidates.head(top_k).iterrows():

            result = {
                "model_id": str(row["model_id"]),
                "brand": row["brand"],
                "model": row["model"],
                "price_min": row["price_min"],
                "price_max": row["price_max"],
                "storage_options": self._parse_list(
                    row["storage_options"]
                ),
                "ram_options": self._parse_list(
                    row["ram_options"]
                ),
                "network": self._parse_list(
                    row["network"]
                ),
                "category": row["category"],
                "in_stock": row["in_stock"],
                "semantic_score": float(
                    row["semantic_score"]
                ),
                "spec_text": row["spec_text"],
            }

            results.append(result)

        logger.info(
            f"Hybrid retrieval returned {len(results)} phones."
        )

        return results

    # ============================================================
    # Cosine similarity
    # ============================================================

    @staticmethod
    def _cosine_similarity(
        query_embedding,
        embeddings,
    ):
        """
        Calculate cosine similarity between one query embedding
        and multiple phone embeddings.
        """

        import numpy as np

        query_embedding = np.asarray(
            query_embedding,
            dtype=float,
        )

        embeddings = np.asarray(
            embeddings,
            dtype=float,
        )

        query_norm = np.linalg.norm(
            query_embedding
        )

        embedding_norms = np.linalg.norm(
            embeddings,
            axis=1,
        )

        # Avoid division by zero
        if query_norm == 0:
            return np.zeros(len(embeddings))

        embedding_norms = np.where(
            embedding_norms == 0,
            1,
            embedding_norms,
        )

        scores = np.dot(
            embeddings,
            query_embedding,
        ) / (
            embedding_norms * query_norm
        )

        return scores
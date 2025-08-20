import json
import os
from typing import Dict, List

import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from src.config import Config
from src.embedding_generator import EmbeddingGenerator


class SemanticSearchEngine:
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.embeddings_data = self._load_embeddings()

        if self.embeddings_data:
            # Get dimension from the first valid record
            first_embedding = self.embeddings_data[0].get("embedding", [])
            correct_dim = len(first_embedding)
            self.embedding_generator.embedding_dim = correct_dim
            print(f"Search engine initialized with embedding dimension: {correct_dim}")

    def _load_embeddings(self):
        path = Config.EMBEDDINGS_PATH
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            st.warning("No embeddings found. Please generate them first.")
            return []
        with open(path, "r") as f:
            try:
                data = json.load(f)
                if isinstance(data, dict) and "embeddings" in data:
                    loaded_embeddings = data["embeddings"]
                else:
                    loaded_embeddings = data
                if not loaded_embeddings:
                    return []
                expected_dim = len(loaded_embeddings[0].get("embedding", []))
                if expected_dim == 0:
                    st.error(
                        "The first embedding record is empty. Cannot validate data."
                    )
                    return []
                validated_embeddings = []
                for item in loaded_embeddings:
                    embedding_len = len(item.get("embedding", []))
                    if embedding_len == expected_dim:
                        validated_embeddings.append(item)
                    else:
                        st.warning(
                            f"⚠️ Skipping record for '{item.get('participant', 'Unknown')}' due to inconsistent embedding dimension (Expected: {expected_dim}, Found: {embedding_len})."
                        )
                return validated_embeddings
            except (json.JSONDecodeError, IndexError) as e:
                st.error(f"Invalid or empty embeddings file: {e}")
                return []

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for similar insights using semantic similarity."""

        if not self.embeddings_data:
            return []
        query_embedding = self.embedding_generator.generate_embedding(query)
        query_vector = np.array(query_embedding).reshape(1, -1)
        stored_embeddings = []
        for item in self.embeddings_data:
            stored_embeddings.append(item["embedding"])
        stored_matrix = np.array(stored_embeddings)
        similarities = cosine_similarity(query_vector, stored_matrix)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            similarity_score = float(similarities[idx])
            result = {
                "participant": self.embeddings_data[idx]["participant"],
                "similarity_score": similarity_score,
                "searchable_text": self.embeddings_data[idx]["searchable_text"],
                "metadata": self.embeddings_data[idx]["metadata"],
                "explanation": self._explain_match(query, self.embeddings_data[idx]),
            }
            results.append(result)
        return results

    def _explain_match(self, query: str, matched_item: Dict) -> str:
        """Generate human-readable explanation of why this matched."""
        metadata = matched_item["metadata"]
        explanations = []
        if any(
            word in metadata.get("primary_goal", "").lower()
            for word in query.lower().split()
        ):
            explanations.append("Similar goal")
        if any(
            word in metadata.get("main_blocker", "").lower()
            for word in query.lower().split()
        ):
            explanations.append("Similar challenge")
        if metadata.get("business_focus", "").lower() in query.lower():
            explanations.append("Same business type")
        if not explanations:
            explanations.append("Semantic similarity")
        return " + ".join(explanations)

import json
from typing import Dict, List

import google.generativeai as genai

from src.config import Config


class EmbeddingGenerator:
    def __init__(self):
        self.embedding_dim = 768
        try:
            genai.configure(api_key=Config.GOOGLE_API_KEY)
            self.api_available = self._check_api_availability()
        except Exception as e:
            print(f"Gemini API configuration failed: {e}")
            self.api_available = False

    def _check_api_availability(self) -> bool:
        """Check if the Gemini embedding API is available."""
        try:
            genai.embed_content(model=Config.GEMINI_EMBEDDING_MODEL, content="test")
            print("Gemini Embedding API is available.")
            return True
        except Exception as e:
            print(f"Gemini Embedding API not available: {e}")
            return False

    def create_searchable_text(self, insight: Dict) -> str:
        """Combine insights into searchable text."""
        searchable_parts = [
            f"Goal: {insight.get('primary_goal', '')}",
            f"Main challenge: {insight.get('main_blocker', '')}",
            f"Business: {insight.get('business_focus', '')}",
            f"Mindset: {insight.get('mindset_pattern', '')}",
            f"Current stage: {insight.get('current_stage', '')}",
        ]
        secondary = insight.get("secondary_blockers", [])
        if secondary:
            searchable_parts.append(f"Other challenges: {', '.join(secondary)}")
        emotions = insight.get("key_emotions", [])
        if emotions:
            searchable_parts.append(f"Emotions: {', '.join(emotions)}")
        return " | ".join(searchable_parts)

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Gemini."""
        if not self.api_available:
            return [0.0] * self.embedding_dim

        try:
            result = genai.embed_content(
                model=Config.GEMINI_EMBEDDING_MODEL, content=text
            )
            return result["embedding"]
        except Exception as e:
            print(f"Error generating embedding with Gemini: {e}")
            return [0.0] * self.embedding_dim

    def process_all_insights(self) -> List[Dict]:
        """Generate embeddings using Gemini or local fallback."""
        if not self.api_available:
            print("🎭 Using existing embeddings file as a fallback.")
            with open(Config.EMBEDDINGS_PATH, "r") as f:
                return json.load(f)

        with open(Config.INSIGHTS_PATH, "r") as f:
            insights = json.load(f)

        results = []
        for insight in insights:
            searchable_text = self.create_searchable_text(insight)
            embedding = self.generate_embedding(searchable_text)
            result = {
                "id": insight["id"],
                "participant": insight["participant"],
                "searchable_text": searchable_text,
                "embedding": embedding,
                "metadata": {
                    "primary_goal": insight.get("primary_goal"),
                    "main_blocker": insight.get("main_blocker"),
                    "business_focus": insight.get("business_focus"),
                    "mindset_pattern": insight.get("mindset_pattern"),
                    "urgency_level": insight.get("urgency_level"),
                },
            }
            results.append(result)

        with open(Config.EMBEDDINGS_PATH, "w") as f:
            json.dump(results, f, indent=2)

        return results

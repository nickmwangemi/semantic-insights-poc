import json
from typing import Dict, List

import google.generativeai as genai

from src.config import Config


class TranscriptProcessor:
    def __init__(self):
        try:
            genai.configure(api_key=Config.GOOGLE_API_KEY)
            self.client = genai.GenerativeModel(Config.GEMINI_GENERATIVE_MODEL)
            self.api_available = self._check_api_availability()
        except Exception as e:
            print(f"Gemini API configuration failed: {e}")
            self.client = None
            self.api_available = False

    def _check_api_availability(self) -> bool:
        """Check if the Gemini API is available."""
        if not self.client:
            return False
        try:
            # A simple, low-cost check
            self.client.generate_content(
                "test", generation_config={"max_output_tokens": 5}
            )
            print("Gemini API is available.")
            return True
        except Exception as e:
            print(f"Gemini API not available: {e}")
            return False

    def _load_local_insights(self) -> List[Dict]:
        """Load insights from the local JSON file as a fallback."""
        with open(Config.INSIGHTS_PATH, "r") as f:
            return json.load(f)

    def process_all_transcripts(self) -> List[Dict]:
        """Process all transcripts using Gemini or local fallback."""
        if not self.api_available:
            print("🎭 Using local insights file as a fallback.")
            return self._load_local_insights()

        with open(Config.TRANSCRIPTS_PATH, "r") as f:
            transcripts = json.load(f)

        results = []
        for transcript_data in transcripts:
            print(f"Processing transcript {transcript_data['id']} with Gemini...")
            prompt = f"""
            Analyze this coaching session transcript and extract key insights. Return your response as a valid JSON object with exactly these fields:
            {{
                "primary_goal": "Main business/personal goal (be specific with numbers if mentioned)",
                "main_blocker": "Primary obstacle or challenge preventing progress",
                "secondary_blockers": ["List of additional challenges"],
                "business_focus": "Industry or business type (e.g., 'e-commerce', 'saas', 'coaching')",
                "mindset_pattern": "Dominant psychological pattern or limiting belief",
                "current_stage": "Where they are now (revenue, team size, etc.)",
                "key_emotions": ["Primary emotions expressed"],
                "urgency_level": "1-5 scale of how urgent their situation feels"
            }}
            
            Transcript: {transcript_data['transcript']}
            """
            try:
                response = self.client.generate_content(prompt)
                # Clean the response to ensure it is valid JSON
                cleaned_response = (
                    response.text.strip().replace("```json", "").replace("```", "")
                )
                insights = json.loads(cleaned_response)

                result = {
                    "id": transcript_data["id"],
                    "participant": transcript_data["participant"],
                    **insights,
                }
                results.append(result)

            except Exception as e:
                print(f"Error processing transcript with Gemini: {e}")
                continue  # Skip this transcript on error

        with open(Config.INSIGHTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

        return results

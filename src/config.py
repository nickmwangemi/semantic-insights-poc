import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    GEMINI_GENERATIVE_MODEL = "gemini-1.5-flash"  # For insight extraction
    GEMINI_EMBEDDING_MODEL = "models/embedding-001"  # For semantic embeddings

    TRANSCRIPTS_PATH = "data/sample_transcripts.json"
    INSIGHTS_PATH = "data/extracted_insights.json"
    EMBEDDINGS_PATH = "data/embeddings.json"

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Google AI (Gemini) Configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GEMINI_GENERATIVE_MODEL = "gemini-1.5-flash"  # For insight extraction
    GEMINI_EMBEDDING_MODEL = "models/embedding-001"  # For semantic embeddings

    # Pinecone Configuration (Optional)
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "semantic-insights")

    # OpenAI Configuration (Alternative embedding provider)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

    # File Paths
    TRANSCRIPTS_PATH = "data/sample_transcripts.json"
    INSIGHTS_PATH = "data/extracted_insights.json"
    EMBEDDINGS_PATH = "data/embeddings.json"

    # Application Settings
    DEFAULT_EMBEDDING_DIMENSION = 768
    MAX_SEARCH_RESULTS = 20
    DEFAULT_SEARCH_RESULTS = 5

    # Vector Store Settings
    USE_PINECONE = os.getenv("USE_PINECONE", "false").lower() == "true"
    PINECONE_BATCH_SIZE = 100

    # Search Configuration
    SIMILARITY_THRESHOLD = 0.7
    MAX_METADATA_LENGTH = 1000

    @classmethod
    def validate_config(cls) -> dict:
        """Validate configuration and return status."""
        status = {
            "gemini_api": bool(cls.GOOGLE_API_KEY),
            "pinecone_api": bool(cls.PINECONE_API_KEY),
            "openai_api": bool(cls.OPENAI_API_KEY),
            "use_pinecone": cls.USE_PINECONE and bool(cls.PINECONE_API_KEY),
            "vector_store": "pinecone" if (cls.USE_PINECONE and cls.PINECONE_API_KEY) else "local"
        }
        return status

    @classmethod
    def get_embedding_config(cls) -> dict:
        """Get embedding configuration based on available APIs."""
        if cls.GOOGLE_API_KEY:
            return {
                "provider": "gemini",
                "model": cls.GEMINI_EMBEDDING_MODEL,
                "dimension": cls.DEFAULT_EMBEDDING_DIMENSION
            }
        elif cls.OPENAI_API_KEY:
            return {
                "provider": "openai",
                "model": cls.OPENAI_EMBEDDING_MODEL,
                "dimension": 1536  # OpenAI text-embedding-3-small dimension
            }
        else:
            return {
                "provider": "local",
                "model": "dummy",
                "dimension": cls.DEFAULT_EMBEDDING_DIMENSION
            }
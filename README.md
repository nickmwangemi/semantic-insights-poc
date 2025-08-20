# Semantic Insights PoC

AI-powered insight extraction and semantic search for coaching session transcripts. Uses Google Gemini API for processing and provides interactive search via Streamlit.

## Features

- AI insight extraction from raw transcripts
- Semantic search across participants
- Interactive dashboard
- Real-time processing
- Demo mode fallback

## Architecture

```
semantic-insights-poc/
├── app.py                      # Streamlit application
├── requirements.txt            
├── data/
│   ├── sample_transcripts.json 
│   ├── extracted_insights.json 
│   └── embeddings.json         
└── src/
    ├── config.py              
    ├── transcript_processor.py 
    ├── embedding_generator.py  
    ├── search_engine.py       
    └── vector_store.py        
```

## Setup

### Requirements
- Python 3.8+
- Google AI API key
- Optional: Pinecone API key

### Installation

```bash
git clone <repository-url>
cd semantic-insights-poc
pip install -r requirements.txt
```

### Configuration

Create `.env` file:
```env
GOOGLE_API_KEY=your_gemini_api_key

# Optional
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=us-east-1
USE_PINECONE=true
OPENAI_API_KEY=your_openai_key
```

### Run

```bash
# Quick setup
python run_vector_store.py --setup

# Or Streamlit interface
streamlit run app.py
```

## Vector Store Commands

```bash
python run_vector_store.py --setup    # Complete setup
python run_vector_store.py --test     # Test existing
python run_vector_store.py --search   # Interactive search
python run_vector_store.py --status   # Check status
```

## Usage

1. Process transcripts to extract insights
2. Generate embeddings for semantic search
3. Search for patterns and similar challenges

### Example Searches
- "delegation and team management issues"
- "pricing confidence problems"
- "e-commerce SaaS participants"

## Data Formats

### Transcript
```json
{
  "id": "transcript_001",
  "participant": "Sarah Chen",
  "session_date": "2024-01-15",
  "transcript": "Raw text..."
}
```

### Extracted Insights
```json
{
  "id": "transcript_001",
  "participant": "Sarah Chen",
  "primary_goal": "Scale revenue",
  "main_blocker": "Delegation issues",
  "secondary_blockers": ["Quality control"],
  "business_focus": "e-commerce",
  "mindset_pattern": "Perfectionist",
  "current_stage": "2 years, $50K/month",
  "key_emotions": ["overwhelmed"],
  "urgency_level": 4
}
```

## Search Engine

Supports local storage and Pinecone vector database.

### Python Integration
```python
from src.vector_store import VectorStore, EnhancedSemanticSearchEngine

vector_store = VectorStore(use_pinecone=True)
search_engine = EnhancedSemanticSearchEngine(use_pinecone=True)
results = search_engine.search("delegation challenges", top_k=5)
```

### Advanced Search
```python
# Filter by business type
results = search_engine.search_with_business_filter(
    query="delegation",
    business_types=["e-commerce"],
    top_k=5
)

# High-urgency search
urgent = search_engine.search_by_urgency(
    query="time management",
    min_urgency=4
)
```

## Pinecone Production Setup

```bash
# Set environment variables
USE_PINECONE=true
PINECONE_API_KEY=your_key

# Migrate from local to Pinecone
python run_vector_store.py --setup
```

## Models

- Insight Extraction: `gemini-1.5-flash`
- Embeddings: `models/embedding-001` (768 dimensions)

## Dependencies

- streamlit
- google-generativeai
- scikit-learn
- numpy
- pandas
- python-dotenv
- pinecone-client (optional)
- openai (optional)

## Improvements Needed

1. **Vector Databases**: Add Weaviate, Qdrant, Chroma support
2. **Search**: Hybrid search, boolean operators
3. **Insights**: Schema validation, confidence scoring
4. **UX**: Real-time search, visual analytics, export
5. **Performance**: Caching, batch processing
6. **Security**: Encryption, access control, audit logs
7. **Production**: Error handling, monitoring, CI/CD
8. **Analytics**: Pattern recognition, recommendations, clustering
9. **Multi-modal**: Audio, documents, images
10. **API**: REST API, webhooks, SDKs

## License

MIT
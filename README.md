# 🔍 Semantic Insights PoC

A proof-of-concept AI application for extracting and searching insights from coaching transcripts. It processes raw text, extracts structured data with Gemini, generates embeddings, and provides a semantic search interface via Streamlit.

**Key Features**: AI insight extraction, semantic search, interactive dashboard, real-time processing, and a local data fallback mode.

## 🏗️ Architecture

```
semantic-insights-poc/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── data/
│   ├── sample_transcripts.json # Raw coaching session transcripts
│   ├── extracted_insights.json # AI-extracted structured insights
│   └── embeddings.json         # Generated embeddings for search
└── src/
    ├── config.py              # Configuration and environment variables
    ├── transcript_processor.py # AI insight extraction logic
    ├── embedding_generator.py  # Embedding generation with Gemini
    ├── search_engine.py       # Semantic search implementation
    └── vector_store.py        # Pinecone & local vector storage
```

## 🚀 Setup & Installation

**Prerequisites**: Python 3.8+, Google AI API key. Optional: Pinecone API key.

1.  **Clone & Install**

    ```bash
    git clone <repository-url>
    cd semantic-insights-poc
    pip install -r requirements.txt
    ```

2.  **Configure Environment**
    Create a `.env` file with your API keys:

    ```env
    # Required
    GOOGLE_API_KEY="your_gemini_api_key"

    # Optional
    PINECONE_API_KEY="your_pinecone_api_key"
    USE_PINECONE=true
    OPENAI_API_KEY="your_openai_api_key"
    ```

3.  **Run Application**

    ```bash
    streamlit run app.py
    ```

## 🚀 Running the Vector Store

The project includes a script for vector store management.

**Automated Setup (Recommended)**:
This command validates APIs, processes transcripts, generates embeddings, and initializes the vector store (Pinecone or local).

```bash
python run_vector_store.py --setup
```

**Other Commands**:

```bash
# Test connection, search, or check status
python run_vector_store.py --test
python run_vector_store.py --search
python run_vector_store.py --status
```

**Manual Setup**: Alternatively, use the sidebar controls in the Streamlit app (`streamlit run app.py`) to process data and generate embeddings.

## 🎮 Usage

1.  **Process Transcripts**: Click "📄 Process Transcripts" to extract insights.
2.  **Generate Embeddings**: Click "🧠 Generate Embeddings" to create vectors.
3.  **Search**: Use the search bar to find patterns.

**Example Searches**:

  * "Who struggles with delegation?"
  * "Find others with pricing and confidence issues"
  * "Show me people in e-commerce or SaaS"

## 🔧 Configuration

Configure via `.env` file. `GOOGLE_API_KEY` is required.

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `GOOGLE_API_KEY` | Google AI (Gemini) API key | Yes | - |
| `PINECONE_API_KEY` | Pinecone vector database API key | No | - |
| `PINECONE_ENVIRONMENT` | Pinecone environment (e.g., 'us-east-1') | No | us-east-1 |
| `PINECONE_INDEX_NAME` | Pinecone index name | No | semantic-insights |
| `USE_PINECONE` | Enable Pinecone vector database | No | false |
| `OPENAI_API_KEY` | OpenAI API key (alternative) | No | - |

**Models**: `gemini-1.5-flash` for extraction, `models/embedding-001` for 768-dim embeddings.

## 📊 Data Structure

### Transcript Format

```json
{
  "id": "transcript_001",
  "participant": "Sarah Chen",
  "transcript": "Raw coaching session text..."
}
```

### Extracted Insights Format

```json
{
  "id": "transcript_001",
  "participant": "Sarah Chen",
  "primary_goal": "Scale from $50K to $200K monthly revenue",
  "main_blocker": "Unable to delegate effectively",
  "business_focus": "e-commerce",
  "urgency_level": 4
}
```

## 🔍 Search Engine

Supports two modes for semantic search:

  * **Local Storage (Default)**: Uses cosine similarity on local files. Ideal for development.
  * **Pinecone (Production)**: Scalable vector database with metadata filtering. Enable with `USE_PINECONE=true`.

**Process**: Query -\> Embed -\> Vector Search -\> Filter -\> Rank.

**Advanced Search Example**:

```python
from src.vector_store import EnhancedSemanticSearchEngine
search_engine = EnhancedSemanticSearchEngine(use_pinecone=True)

# Filter by business type
results = search_engine.search_with_business_filter(
    query="delegation issues",
    business_types=["e-commerce", "saas"], top_k=5
)
# Find similar participants
similar = search_engine.get_similar_participants("Sarah Chen", top_k=5)
```

## 🎛️ API Integration

  * **Google Gemini**: Used for structured insight extraction and generating semantic embeddings.
  * **Pinecone**: An optional, scalable vector database for production. The system auto-initializes the index.
  * **OpenAI**: Supported as an alternative for embedding generation.

**Pinecone Example**:

```python
from src.vector_store import VectorStore
# Initialize with Pinecone (auto-creates index)
vs = VectorStore(use_pinecone=True)
# Store embeddings
vs.upsert_embeddings(embeddings_data)
# Search with metadata filtering
results = vs.search(query_embedding, top_k=10, filters={'urgency_level': 4})
```

## 🔧 Things to Improve

  * **Vector Database**: Add support for Weaviate, Qdrant, Chroma.
  * **Search**: Implement hybrid (semantic + keyword) search and advanced query operators.
  * **Insight Extraction**: Add schema validation and confidence scoring.
  * **UX**: Implement search-as-you-type, visual analytics, and data export.
  * **Performance**: Add caching and lazy loading.
  * **Security**: Implement data encryption and access control.
  * **Production**: Improve error handling, monitoring, and testing (CI/CD).
  * **Analytics**: Add trend analysis, a recommendation engine, and clustering.
  * **Multi-modal**: Support direct audio processing and document analysis.
  * **API**: Develop a REST API with webhooks for programmatic access.

## 📦 Dependencies

  * `streamlit`
  * `google-generativeai`
  * `scikit-learn`
  * `numpy`
  * `pandas`
  * `python-dotenv`
  * `pinecone-client` (optional)
  * `openai` (optional)

## 🤝 Contributing

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/amazing-feature`).
3.  Commit your changes (`git commit -m 'Add amazing feature'`).
4.  Push to the branch (`git push origin feature/amazing-feature`).
5.  Open a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

  * Google AI for the Gemini API
  * Streamlit for the web framework

-----

## 🚀 Production Deployment

For production, use Pinecone for scalability.

1.  **Configure**: Set `USE_PINECONE=true` and add `PINECONE_API_KEY` to your `.env` file.
2.  **Migrate Data**: Use the provided logic to upsert existing embeddings into your Pinecone index.
    ```python
    from src.vector_store import VectorStore
    # Ensure embeddings are generated
    pinecone_vs = VectorStore(use_pinecone=True)
    success = pinecone_vs.upsert_embeddings(embeddings)
    if success: print("✅ Migrated to Pinecone!")
    ```
3.  **Monitor**: Check the health and stats of your vector store.
    ```python
    stats = pinecone_vs.get_stats()
    print(f"Backend: {stats['backend']}, Total vectors: {stats['total_vectors']}")
    ```

*Note: This is a PoC. Ensure data privacy compliance for production use.*
# 🔍 Semantic Insights PoC

A proof-of-concept application that demonstrates AI-powered insight extraction and semantic search capabilities for coaching session transcripts. The system processes raw transcripts, extracts structured insights using AI, generates semantic embeddings, and provides intelligent search functionality.

## 🎯 Overview

This application showcases how AI can transform unstructured coaching conversations into searchable, actionable insights. It uses Google's Gemini API for insight extraction and embedding generation, with a Streamlit interface for interactive exploration.

### Key Features

- **🤖 AI-Powered Insight Extraction**: Uses Gemini to extract structured insights from raw transcripts
- **🔍 Semantic Search**: Find similar challenges, goals, and patterns across participants
- **📊 Interactive Dashboard**: Visual overview of insights and participant profiles
- **⚡ Real-time Processing**: Process transcripts and generate embeddings on-demand
- **🎭 Demo Mode**: Fallback to local data when API is unavailable

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

### Prerequisites

- Python 3.8+
- Google AI API key (Gemini)
- Optional: Pinecone API key for vector database

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd semantic-insights-poc
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   Create a `.env` file in the root directory:
   ```env
   # Required: Google AI API
   GOOGLE_API_KEY=your_gemini_api_key_here
   
   # Optional: Pinecone Vector Database (Recommended for production)
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_ENVIRONMENT=us-east-1
   USE_PINECONE=true
   
   # Optional: Alternative embedding provider
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🚀 Running the Vector Store

### Quick Start Commands

The project includes a standalone script for easy vector store setup and management:

```bash
# Complete setup (recommended for first time)
python run_vector_store.py --setup

# Test existing setup
python run_vector_store.py --test

# Interactive search interface
python run_vector_store.py --search

# Check system status
python run_vector_store.py --status
```

### Setup Options

#### **Option 1: Automated Setup Script**
```bash
# Download and save the run_vector_store.py script to your project root
python run_vector_store.py --setup
```

This command will:
- ✅ Validate your API configuration
- 📋 Process transcripts using Gemini AI
- 🧠 Generate semantic embeddings
- 💾 Initialize vector store (Pinecone or local fallback)
- 🔍 Test search functionality
- 📊 Display setup statistics

#### **Option 2: Manual Streamlit Setup**
```bash
streamlit run app.py
```

Then use the sidebar controls:
1. Click "📄 Process Transcripts"
2. Click "🧠 Generate Embeddings" 
3. Start searching in the main interface

#### **Option 3: Python Integration**
```python
from src.vector_store import VectorStore, EnhancedSemanticSearchEngine

# Initialize vector store (auto-detects Pinecone availability)
vector_store = VectorStore(use_pinecone=True)

# Set up search engine
search_engine = EnhancedSemanticSearchEngine(use_pinecone=True)

# Search for insights
results = search_engine.search("delegation challenges", top_k=5)
```

## 🎮 Usage

### Basic Workflow

1. **Process Transcripts**: Click "📄 Process Transcripts" to extract insights from raw data
2. **Generate Embeddings**: Click "🧠 Generate Embeddings" to create semantic vectors
3. **Search & Explore**: Use the search interface to find similar patterns and challenges

### Example Searches

- "Who struggles with delegation and team management?"
- "Find others with pricing and confidence issues"
- "Show me people in e-commerce or SaaS"
- "Who has imposter syndrome or self-doubt?"

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `GOOGLE_API_KEY` | Google AI (Gemini) API key | Yes | - |
| `PINECONE_API_KEY` | Pinecone vector database API key | No | - |
| `PINECONE_ENVIRONMENT` | Pinecone environment (e.g., 'us-east-1') | No | us-east-1 |
| `PINECONE_INDEX_NAME` | Pinecone index name | No | semantic-insights |
| `USE_PINECONE` | Enable Pinecone vector database | No | false |
| `OPENAI_API_KEY` | OpenAI API key (alternative embeddings) | No | - |

### Models Used

- **Insight Extraction**: `gemini-1.5-flash`
- **Embeddings**: `models/embedding-001`
- **Embedding Dimensions**: 768

## 📊 Data Structure

### Transcript Format
```json
{
  "id": "transcript_001",
  "participant": "Sarah Chen",
  "session_date": "2024-01-15",
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
  "secondary_blockers": ["Quality control concerns", "Perfectionist mindset"],
  "business_focus": "e-commerce",
  "mindset_pattern": "Perfectionist holding back growth",
  "current_stage": "2 years, 8 employees, $50K monthly",
  "key_emotions": ["overwhelmed", "worried"],
  "urgency_level": 4
}
```

## 🔍 Search Engine

The semantic search engine supports both local storage and Pinecone vector database:

### Local Storage Mode (Default)
- Uses cosine similarity on locally stored embeddings
- Suitable for development and small datasets
- No additional setup required

### Pinecone Mode (Production)
- Scalable vector database for large datasets
- Sub-second search performance
- Advanced metadata filtering
- Real-time updates

### Search Process
1. **Query Processing**: User query is embedded using Gemini
2. **Vector Search**: Similarity search in vector database
3. **Metadata Filtering**: Optional filtering by business type, urgency, etc.
4. **Ranking**: Results ranked by similarity score
5. **Explanation**: Human-readable explanation of matches

### Advanced Search Features
```python
from src.vector_store import EnhancedSemanticSearchEngine

search_engine = EnhancedSemanticSearchEngine(use_pinecone=True)

# Filter by business type
results = search_engine.search_with_business_filter(
    query="delegation issues",
    business_types=["e-commerce", "saas"],
    top_k=5
)

# Find high-urgency situations
urgent_results = search_engine.search_by_urgency(
    query="time management",
    min_urgency=4,
    top_k=3
)

# Find similar participants
similar = search_engine.get_similar_participants(
    participant_name="Sarah Chen",
    top_k=5
)
```

## 🎛️ API Integration

### Gemini API
The system integrates with Google's Gemini API for:
- **Text Generation**: Extracting structured insights from transcripts
- **Embeddings**: Creating semantic vectors for search

### Pinecone Vector Database
For production scalability, the system supports Pinecone:

```python
from src.vector_store import VectorStore

# Initialize with Pinecone (auto-creates index if needed)
vector_store = VectorStore(use_pinecone=True)

# Store embeddings in Pinecone
vector_store.upsert_embeddings(embeddings_data)

# Search with metadata filtering
results = vector_store.search(
    query_embedding=query_vector,
    top_k=10,
    filters={
        'business_focus': 'e-commerce',
        'urgency_level': 4
    }
)

# Get vector store statistics
stats = vector_store.get_stats()
print(f"Total vectors: {stats['total_vectors']}")
```

### Pinecone Setup
1. **Create Account**: Sign up at [pinecone.io](https://pinecone.io)
2. **Get API Key**: Generate API key from dashboard
3. **Configure Environment**: Set `PINECONE_API_KEY` and `USE_PINECONE=true`
4. **Auto-Initialization**: The system automatically creates the index

### OpenAI Alternative (Optional)
For embedding generation, you can also use OpenAI:
```env
OPENAI_API_KEY=your_openai_key
```

## 🔧 Things to Improve

### 1. **Vector Database Integration**
- **Current State**: Supports both local JSON files and Pinecone vector database
- **Improvement**: Add support for other vector databases (Weaviate, Qdrant, Chroma)
- **Benefits**: Better performance, metadata filtering, production scalability
- **Implementation**: The `VectorStore` class provides a unified interface

### 2. **Enhanced Search Capabilities**
- **Metadata Filtering**: ✅ Implemented - Filter by business type, urgency level, participant
- **Hybrid Search**: Combine semantic and keyword-based search
- **Advanced Querying**: Support for complex queries with boolean operators
- **Similar Participant Discovery**: ✅ Implemented - Find participants with similar challenges

### 3. **Improved Insight Extraction**
- **Schema Validation**: Validate extracted insights against predefined schemas
- **Confidence Scoring**: Add confidence scores to extracted insights
- **Multi-turn Processing**: Handle longer, more complex coaching sessions

### 4. **User Experience Enhancements**
- **Real-time Search**: Implement search-as-you-type functionality
- **Visual Analytics**: Add charts and graphs for insight visualization
- **Export Functionality**: Allow users to export search results and insights

### 5. **Performance Optimizations**
- **Caching**: Cache embeddings and search results for faster responses
- **Batch Processing**: ✅ Implemented - Process multiple transcripts simultaneously
- **Lazy Loading**: Load embeddings on-demand for large datasets
- **Vector Database**: ✅ Implemented - Pinecone for production-scale performance

### 6. **Security & Privacy**
- **Data Encryption**: Encrypt sensitive transcript data
- **Access Control**: Add user authentication and role-based access
- **Audit Logging**: Track who accesses what data and when

### 7. **Production Readiness**
- **Error Handling**: Robust error handling and recovery mechanisms
- **Monitoring**: Add application monitoring and alerting
- **Testing**: Comprehensive unit and integration tests
- **CI/CD Pipeline**: Automated deployment and testing

### 8. **Advanced Analytics**
- **Pattern Recognition**: Identify trends across coaching sessions over time
- **Recommendation Engine**: Suggest relevant resources or interventions
- **Clustering**: Group similar participants or challenges automatically

### 9. **Multi-modal Support**
- **Audio Processing**: Direct processing of audio recordings
- **Document Analysis**: Support for uploaded documents and notes
- **Image Recognition**: Extract insights from whiteboards or diagrams

### 10. **API Development**
- **REST API**: Provide programmatic access to search and insights
- **Webhooks**: Real-time notifications for new insights or matches
- **SDKs**: Client libraries for different programming languages

## 📦 Dependencies

Core dependencies include:
- `streamlit`: Web application framework
- `google-generativeai`: Google AI API client
- `scikit-learn`: Machine learning utilities
- `numpy`: Numerical computing
- `pandas`: Data manipulation
- `python-dotenv`: Environment variable management
- `pinecone-client`: Pinecone vector database (optional)
- `openai`: OpenAI API client (optional alternative)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google AI for the Gemini API
- Streamlit for the web framework
- The coaching community for inspiring this use case

---

## 🚀 Production Deployment

### Pinecone Setup for Production

1. **Create Pinecone Account**: Sign up at [pinecone.io](https://pinecone.io)
2. **Get API Credentials**: Generate API key and note your environment
3. **Update Environment Variables**:
   ```env
   USE_PINECONE=true
   PINECONE_API_KEY=your_api_key_here
   PINECONE_ENVIRONMENT=us-east-1
   ```
4. **Initialize Vector Store**:
   ```python
   from src.vector_store import VectorStore
   vector_store = VectorStore(use_pinecone=True)
   # Index is automatically created on first run
   ```

### Migration from Local to Pinecone

```python
from src.vector_store import VectorStore
from src.embedding_generator import EmbeddingGenerator

# Generate embeddings if not already done
generator = EmbeddingGenerator()
embeddings = generator.process_all_insights()

# Initialize Pinecone vector store
vector_store = VectorStore(use_pinecone=True)

# Migrate data to Pinecone
success = vector_store.upsert_embeddings(embeddings)
if success:
    print("✅ Successfully migrated to Pinecone!")
    
# Verify migration
stats = vector_store.get_stats()
print(f"Total vectors in Pinecone: {stats['total_vectors']}")
```

### Monitoring and Maintenance

```python
# Check vector store health
stats = vector_store.get_stats()
print(f"Backend: {stats['backend']}")
print(f"Total vectors: {stats['total_vectors']}")

# Clean up old embeddings
vector_store.delete_by_id("old_embedding_id")
```

*This is a proof-of-concept demonstration. For production use, consider the improvements outlined above and ensure compliance with data privacy regulations.*
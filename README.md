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
    └── search_engine.py       # Semantic search implementation
```

## 🚀 Setup & Installation

### Prerequisites

- Python 3.8+
- Google AI API key (Gemini)
- Optional: Pinecone API key for vector database

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone git@github.com:nickmwangemi/semantic-insights-poc.git
   cd semantic-insights-poc
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here  # Optional
   PINECONE_ENVIRONMENT=your_pinecone_env       # Optional
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
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

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google AI (Gemini) API key | Yes |
| `PINECONE_API_KEY` | Pinecone vector database API key | No |
| `PINECONE_ENVIRONMENT` | Pinecone environment (e.g., 'us-west1-gcp') | No |

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

The semantic search engine uses cosine similarity to find relevant matches:

1. **Query Processing**: User query is embedded using Gemini
2. **Similarity Calculation**: Cosine similarity against stored embeddings
3. **Ranking**: Results ranked by similarity score
4. **Explanation**: Human-readable explanation of why results matched

## 🎛️ API Integration

### Gemini API

The system integrates with Google's Gemini API for:
- **Text Generation**: Extracting structured insights from transcripts
- **Embeddings**: Creating semantic vectors for search

### Pinecone Integration (Optional)

For production use, the system supports Pinecone vector database:

```python
# Example Pinecone configuration
import pinecone

pinecone.init(
    api_key="your-api-key",
    environment="your-environment"
)

index = pinecone.Index("semantic-insights")
```

## 🔧 Things to Improve

### 1. **Vector Database Integration**
- **Current State**: Uses local JSON files for embedding storage
- **Improvement**: Integrate with Pinecone or Weaviate for scalable vector search
- **Benefits**: Better performance, metadata filtering, production scalability

### 2. **Enhanced Search Capabilities**
- **Metadata Filtering**: Filter by business type, urgency level, or date ranges
- **Hybrid Search**: Combine semantic and keyword-based search
- **Advanced Querying**: Support for complex queries with boolean operators

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
- **Batch Processing**: Process multiple transcripts simultaneously
- **Lazy Loading**: Load embeddings on-demand for large datasets

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

---

*This is a proof-of-concept demonstration. For production use, consider the improvements outlined above and ensure compliance with data privacy regulations.*
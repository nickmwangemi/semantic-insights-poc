import json
import os
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime
import logging

try:
    import pinecone
    from pinecone import Pinecone, ServerlessSpec

    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logging.warning("Pinecone not installed. Using local storage fallback.")

from src.config import Config


class VectorStore:
    """Unified interface for vector storage - supports both local and Pinecone backends."""

    def __init__(self, use_pinecone: bool = True):
        self.use_pinecone = use_pinecone and PINECONE_AVAILABLE and Config.PINECONE_API_KEY
        self.index_name = "semantic-insights"

        if self.use_pinecone:
            self._init_pinecone()
        else:
            self._init_local_storage()

    def _init_pinecone(self):
        """Initialize Pinecone client and index."""
        try:
            # Initialize Pinecone with new SDK
            pc = Pinecone(api_key=Config.PINECONE_API_KEY)

            # Check if index exists, create if not
            if self.index_name not in pc.list_indexes().names():
                pc.create_index(
                    name=self.index_name,
                    dimension=768,  # Gemini embedding dimension
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=Config.PINECONE_ENVIRONMENT or 'us-east-1'
                    )
                )

            self.index = pc.Index(self.index_name)
            logging.info(f"✅ Pinecone initialized: {self.index_name}")

        except Exception as e:
            logging.error(f"❌ Pinecone initialization failed: {e}")
            self.use_pinecone = False
            self._init_local_storage()

    def _init_local_storage(self):
        """Initialize local JSON storage as fallback."""
        self.local_embeddings_path = Config.EMBEDDINGS_PATH
        logging.info("📁 Using local storage for embeddings")

    def upsert_embeddings(self, embeddings_data: List[Dict]) -> bool:
        """Store embeddings in the vector database."""
        if self.use_pinecone:
            return self._upsert_pinecone(embeddings_data)
        else:
            return self._upsert_local(embeddings_data)

    def _upsert_pinecone(self, embeddings_data: List[Dict]) -> bool:
        """Store embeddings in Pinecone."""
        try:
            vectors = []
            for item in embeddings_data:
                # Prepare metadata (Pinecone has size limits)
                metadata = {
                    "participant": item["participant"],
                    "primary_goal": item["metadata"].get("primary_goal", "")[:500],  # Truncate long text
                    "main_blocker": item["metadata"].get("main_blocker", "")[:500],
                    "business_focus": item["metadata"].get("business_focus", ""),
                    "mindset_pattern": item["metadata"].get("mindset_pattern", "")[:300],
                    "urgency_level": item["metadata"].get("urgency_level", 3),
                    "created_at": datetime.now().isoformat(),
                    "searchable_text": item["searchable_text"][:1000]  # Truncate for metadata
                }

                vectors.append({
                    "id": item["id"],
                    "values": item["embedding"],
                    "metadata": metadata
                })

            # Upsert in batches of 100 (Pinecone limit)
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)

            logging.info(f"✅ Stored {len(vectors)} embeddings in Pinecone")
            return True

        except Exception as e:
            logging.error(f"❌ Error storing embeddings in Pinecone: {e}")
            return False

    def _upsert_local(self, embeddings_data: List[Dict]) -> bool:
        """Store embeddings locally."""
        try:
            with open(self.local_embeddings_path, 'w') as f:
                json.dump(embeddings_data, f, indent=2)
            logging.info(f"✅ Stored {len(embeddings_data)} embeddings locally")
            return True
        except Exception as e:
            logging.error(f"❌ Error storing embeddings locally: {e}")
            return False

    def search(self, query_embedding: List[float], top_k: int = 10,
               filters: Optional[Dict] = None) -> List[Dict]:
        """Search for similar embeddings."""
        if self.use_pinecone:
            return self._search_pinecone(query_embedding, top_k, filters)
        else:
            return self._search_local(query_embedding, top_k, filters)

    def _search_pinecone(self, query_embedding: List[float], top_k: int = 10,
                         filters: Optional[Dict] = None) -> List[Dict]:
        """Search using Pinecone."""
        try:
            # Build Pinecone filter
            pinecone_filter = {}
            if filters:
                if 'business_focus' in filters:
                    pinecone_filter['business_focus'] = filters['business_focus']
                if 'urgency_level' in filters:
                    pinecone_filter['urgency_level'] = {"$gte": filters['urgency_level']}
                if 'participant' in filters:
                    pinecone_filter['participant'] = filters['participant']

            # Perform search
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=pinecone_filter if pinecone_filter else None
            )

            # Format results
            formatted_results = []
            for match in results.matches:
                result = {
                    'id': match.id,
                    'similarity_score': float(match.score),
                    'participant': match.metadata.get('participant', ''),
                    'metadata': {
                        'primary_goal': match.metadata.get('primary_goal', ''),
                        'main_blocker': match.metadata.get('main_blocker', ''),
                        'business_focus': match.metadata.get('business_focus', ''),
                        'mindset_pattern': match.metadata.get('mindset_pattern', ''),
                        'urgency_level': match.metadata.get('urgency_level', 3)
                    },
                    'searchable_text': match.metadata.get('searchable_text', '')
                }
                formatted_results.append(result)

            return formatted_results

        except Exception as e:
            logging.error(f"❌ Error searching Pinecone: {e}")
            return []

    def _search_local(self, query_embedding: List[float], top_k: int = 10,
                      filters: Optional[Dict] = None) -> List[Dict]:
        """Search using local storage."""
        try:
            # Load local embeddings
            if not os.path.exists(self.local_embeddings_path):
                return []

            with open(self.local_embeddings_path, 'r') as f:
                embeddings_data = json.load(f)

            if not embeddings_data:
                return []

            # Apply filters
            if filters:
                filtered_data = []
                for item in embeddings_data:
                    if self._matches_filters(item, filters):
                        filtered_data.append(item)
                embeddings_data = filtered_data

            # Calculate similarities
            query_vector = np.array(query_embedding).reshape(1, -1)
            similarities = []

            for item in embeddings_data:
                embedding_vector = np.array(item['embedding']).reshape(1, -1)
                similarity = np.dot(query_vector, embedding_vector.T)[0][0]
                similarities.append((similarity, item))

            # Sort by similarity and get top_k
            similarities.sort(key=lambda x: x[0], reverse=True)
            results = []

            for similarity, item in similarities[:top_k]:
                result = {
                    'id': item['id'],
                    'similarity_score': float(similarity),
                    'participant': item['participant'],
                    'metadata': item['metadata'],
                    'searchable_text': item['searchable_text']
                }
                results.append(result)

            return results

        except Exception as e:
            logging.error(f"❌ Error searching locally: {e}")
            return []

    def _matches_filters(self, item: Dict, filters: Dict) -> bool:
        """Check if an item matches the given filters."""
        metadata = item.get('metadata', {})

        if 'business_focus' in filters:
            if metadata.get('business_focus', '').lower() != filters['business_focus'].lower():
                return False

        if 'urgency_level' in filters:
            item_urgency = metadata.get('urgency_level', 3)
            if item_urgency < filters['urgency_level']:
                return False

        if 'participant' in filters:
            if item.get('participant', '').lower() != filters['participant'].lower():
                return False

        return True

    def delete_by_id(self, embedding_id: str) -> bool:
        """Delete an embedding by ID."""
        if self.use_pinecone:
            try:
                self.index.delete(ids=[embedding_id])
                logging.info(f"✅ Deleted embedding {embedding_id} from Pinecone")
                return True
            except Exception as e:
                logging.error(f"❌ Error deleting from Pinecone: {e}")
                return False
        else:
            try:
                with open(self.local_embeddings_path, 'r') as f:
                    embeddings_data = json.load(f)

                # Filter out the item to delete
                embeddings_data = [item for item in embeddings_data if item['id'] != embedding_id]

                with open(self.local_embeddings_path, 'w') as f:
                    json.dump(embeddings_data, f, indent=2)

                logging.info(f"✅ Deleted embedding {embedding_id} from local storage")
                return True

            except Exception as e:
                logging.error(f"❌ Error deleting locally: {e}")
                return False

    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        if self.use_pinecone:
            try:
                stats = self.index.describe_index_stats()
                return {
                    'total_vectors': stats.total_vector_count,
                    'backend': 'pinecone',
                    'index_name': self.index_name,
                    'dimension': 768
                }
            except Exception as e:
                logging.error(f"❌ Error getting Pinecone stats: {e}")
                return {'error': str(e)}
        else:
            try:
                if os.path.exists(self.local_embeddings_path):
                    with open(self.local_embeddings_path, 'r') as f:
                        embeddings_data = json.load(f)

                    return {
                        'total_vectors': len(embeddings_data),
                        'backend': 'local',
                        'file_path': self.local_embeddings_path,
                        'dimension': len(embeddings_data[0]['embedding']) if embeddings_data else 0
                    }
                else:
                    return {
                        'total_vectors': 0,
                        'backend': 'local',
                        'file_path': self.local_embeddings_path,
                        'dimension': 0
                    }
            except Exception as e:
                logging.error(f"❌ Error getting local stats: {e}")
                return {'error': str(e)}


# Enhanced Search Engine with Pinecone support
class EnhancedSemanticSearchEngine:
    """Enhanced search engine with Pinecone integration and advanced filtering."""

    def __init__(self, use_pinecone: bool = True):
        from src.embedding_generator import EmbeddingGenerator

        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore(use_pinecone=use_pinecone)

    def search(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """Enhanced search with filtering capabilities."""
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)

        # Search using vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters
        )

        # Add explanations to results
        for result in results:
            result['explanation'] = self._explain_match(query, result)

        return results

    def _explain_match(self, query: str, matched_item: Dict) -> str:
        """Generate human-readable explanation of why this matched."""
        metadata = matched_item["metadata"]
        explanations = []

        query_lower = query.lower()

        # Check for goal similarity
        if any(word in metadata.get("primary_goal", "").lower() for word in query_lower.split()):
            explanations.append("Similar goal")

        # Check for challenge similarity
        if any(word in metadata.get("main_blocker", "").lower() for word in query_lower.split()):
            explanations.append("Similar challenge")

        # Check for business type match
        if metadata.get("business_focus", "").lower() in query_lower:
            explanations.append("Same business type")

        # Check for mindset patterns
        if any(word in metadata.get("mindset_pattern", "").lower() for word in query_lower.split()):
            explanations.append("Similar mindset")

        if not explanations:
            explanations.append("Semantic similarity")

        return " + ".join(explanations)

    def search_with_business_filter(self, query: str, business_types: List[str], top_k: int = 5) -> List[Dict]:
        """Search within specific business types."""
        all_results = []

        for business_type in business_types:
            filters = {'business_focus': business_type}
            results = self.search(query, top_k, filters)
            all_results.extend(results)

        # Sort by similarity and return top_k
        all_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return all_results[:top_k]

    def search_by_urgency(self, query: str, min_urgency: int = 3, top_k: int = 5) -> List[Dict]:
        """Search for high-urgency situations."""
        filters = {'urgency_level': min_urgency}
        return self.search(query, top_k, filters)

    def get_similar_participants(self, participant_name: str, top_k: int = 5) -> List[Dict]:
        """Find participants with similar challenges/goals."""
        # First, get the participant's data
        filters = {'participant': participant_name}
        participant_data = self.search("", top_k=1, filters=filters)

        if not participant_data:
            return []

        # Use their challenges and goals as the search query
        participant = participant_data[0]
        metadata = participant['metadata']

        search_query = f"{metadata.get('primary_goal', '')} {metadata.get('main_blocker', '')}"

        # Search for similar participants (excluding the original)
        results = self.search(search_query, top_k + 1)
        return [r for r in results if r['participant'] != participant_name][:top_k]
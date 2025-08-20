"""
Standalone script to set up and test the vector store.
Run this to quickly get your vector store up and running.

Usage:
    python run_vector_store.py --setup    # Complete setup
    python run_vector_store.py --test     # Test existing setup
    python run_vector_store.py --search   # Interactive search
    python run_vector_store.py --status   # Check status
"""

import argparse
import os
import sys
from pathlib import Path

from src.config import Config
from src.embedding_generator import EmbeddingGenerator
from src.transcript_processor import TranscriptProcessor
from src.vector_store import EnhancedSemanticSearchEngine, VectorStore

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))


def setup_vector_store():
    """Complete vector store setup"""
    print("🚀 Setting up Vector Store...")
    print("=" * 50)

    try:
        # Step 1: Check configuration
        print("1️⃣  Checking configuration...")

        if not Config.GOOGLE_API_KEY:
            print("❌ GOOGLE_API_KEY not found in environment")
            print("💡 Create a .env file with: GOOGLE_API_KEY=your_key_here")
            return False

        config_status = Config.validate_config()
        print(f"   ✅ Gemini API: {'✓' if config_status['gemini_api'] else '✗'}")
        print(f"   ✅ Pinecone API: {'✓' if config_status['pinecone_api'] else '✗'}")
        print(f"   ✅ Vector Store: {config_status['vector_store']}")

        # Step 2: Process transcripts
        print("\n2️⃣  Processing transcripts...")

        processor = TranscriptProcessor()
        insights = processor.process_all_transcripts()
        print(f"   ✅ Processed {len(insights)} transcripts")

        if insights:
            sample = insights[0]
            print(f"   📋 Sample: {sample['participant']} - {sample.get('primary_goal', 'N/A')[:50]}...")

        # Step 3: Generate embeddings
        print("\n3️⃣  Generating embeddings...")

        generator = EmbeddingGenerator()
        embeddings_data = generator.process_all_insights()
        print(f"   ✅ Generated {len(embeddings_data)} embeddings")

        if embeddings_data:
            embedding_dim = len(embeddings_data[0]['embedding'])
            print(f"   📊 Embedding dimension: {embedding_dim}")

        # Step 4: Initialize vector store
        print("\n4️⃣  Setting up vector store...")

        # Try Pinecone first, fall back to local
        vector_store = VectorStore(use_pinecone=True)
        backend = "Pinecone" if vector_store.use_pinecone else "Local"
        print(f"   ✅ Using {backend} vector store")

        # Store embeddings
        success = vector_store.upsert_embeddings(embeddings_data)
        if success:
            print("   ✅ Embeddings stored successfully")
        else:
            print("   ❌ Error storing embeddings")
            return False

        # Step 5: Verify setup
        print("\n5️⃣  Verifying setup...")
        stats = vector_store.get_stats()
        print(f"   📊 Backend: {stats.get('backend', 'Unknown')}")
        print(f"   📊 Total vectors: {stats.get('total_vectors', 0)}")
        print(f"   📊 Dimensions: {stats.get('dimension', 0)}")

        # Step 6: Test search
        print("\n6️⃣  Testing search...")

        search_engine = EnhancedSemanticSearchEngine(use_pinecone=vector_store.use_pinecone)
        test_results = search_engine.search("delegation issues", top_k=2)

        if test_results:
            print(f"   ✅ Search working! Found {len(test_results)} results")
            result = test_results[0]
            print(f"   📋 Top match: {result['participant']} ({result['similarity_score']:.2%})")
        else:
            print("   ⚠️  Search returned no results")

        print("\n🎉 Setup Complete!")
        print("You can now run: streamlit run app.py")
        return True

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vector_store():
    """Test existing vector store setup"""
    print("🧪 Testing Vector Store...")
    print("=" * 30)

    try:
        # Test configuration
        config_status = Config.validate_config()
        print("📋 Configuration:")
        for key, status in config_status.items():
            print(f"   {'✅' if status else '❌'} {key}")

        # Test vector store
        print("\n💾 Vector Store:")
        vector_store = VectorStore(use_pinecone=True)
        stats = vector_store.get_stats()

        print(f"   Backend: {stats.get('backend', 'Unknown')}")
        print(f"   Vectors: {stats.get('total_vectors', 0)}")
        print(f"   Status: {'✅ Ready' if stats.get('total_vectors', 0) > 0 else '❌ Empty'}")

        # Test search
        print("\n🔍 Search Engine:")
        search_engine = EnhancedSemanticSearchEngine(use_pinecone=vector_store.use_pinecone)

        test_queries = [
            "delegation problems",
            "time management",
            "scaling challenges"
        ]

        for query in test_queries:
            results = search_engine.search(query, top_k=1)
            status = "✅" if results else "❌"
            count = len(results) if results else 0
            print(f"   {status} '{query}': {count} results")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def interactive_search():
    """Interactive search interface"""
    print("🔍 Interactive Search")
    print("=" * 25)
    print("Type 'quit' to exit\n")

    try:
        # Initialize search engine
        search_engine = EnhancedSemanticSearchEngine(use_pinecone=True)
        print("✅ Search engine ready!")

        while True:
            try:
                query = input("\n🔍 Search query: ").strip()

                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break

                if not query:
                    continue

                print(f"Searching for: '{query}'...")
                results = search_engine.search(query, top_k=3)

                if results:
                    print(f"\n📋 Found {len(results)} results:")
                    print("-" * 50)

                    for i, result in enumerate(results, 1):
                        similarity = result['similarity_score']
                        participant = result['participant']
                        goal = result['metadata']['primary_goal']
                        challenge = result['metadata']['main_blocker']
                        business = result['metadata']['business_focus']

                        print(f"{i}. {participant} ({similarity:.1%} match)")
                        print(f"   🏢 Business: {business}")
                        print(f"   🎯 Goal: {goal[:60]}{'...' if len(goal) > 60 else ''}")
                        print(f"   🚫 Challenge: {challenge[:60]}{'...' if len(challenge) > 60 else ''}")
                        print(f"   💡 Why: {result['explanation']}")
                        print()
                else:
                    print("❌ No results found")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Search error: {e}")

    except Exception as e:
        print(f"❌ Failed to initialize search: {e}")


def show_status():
    """Show detailed status of the system"""
    print("📊 System Status")
    print("=" * 20)

    try:
        # Configuration
        from config import Config
        print("🔧 Configuration:")
        config = Config.validate_config()
        for key, value in config.items():
            status = "✅" if value else "❌"
            print(f"   {status} {key.replace('_', ' ').title()}: {value}")

        # Files
        print("\n📁 Data Files:")
        files = [
            ("Transcripts", Config.TRANSCRIPTS_PATH),
            ("Insights", Config.INSIGHTS_PATH),
            ("Embeddings", Config.EMBEDDINGS_PATH)
        ]

        for name, path in files:
            if os.path.exists(path):
                size = os.path.getsize(path)
                print(f"   ✅ {name}: {size:,} bytes")
            else:
                print(f"   ❌ {name}: Not found")

        # Vector store
        print("\n💾 Vector Store:")
        from vector_store import VectorStore

        try:
            vector_store = VectorStore(use_pinecone=True)
            stats = vector_store.get_stats()
            print(f"   Backend: {stats.get('backend', 'Unknown')}")
            print(f"   Vectors: {stats.get('total_vectors', 0):,}")
            print(f"   Dimensions: {stats.get('dimension', 0)}")

            if 'error' in stats:
                print(f"   ❌ Error: {stats['error']}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # Sample data
        print("\n📋 Sample Data:")
        try:
            import json
            with open(Config.INSIGHTS_PATH, 'r') as f:
                insights = json.load(f)

            if insights:
                sample = insights[0]
                print(f"   Participant: {sample.get('participant', 'Unknown')}")
                print(f"   Business: {sample.get('business_focus', 'Unknown')}")
                print(f"   Goal: {sample.get('primary_goal', 'Unknown')[:50]}...")
            else:
                print("   ❌ No insights found")
        except Exception as e:
            print(f"   ❌ Error reading insights: {e}")

    except Exception as e:
        print(f"❌ Status check failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Vector Store Runner")
    parser.add_argument("--setup", action="store_true", help="Complete setup")
    parser.add_argument("--test", action="store_true", help="Test existing setup")
    parser.add_argument("--search", action="store_true", help="Interactive search")
    parser.add_argument("--status", action="store_true", help="Show system status")

    args = parser.parse_args()

    if args.setup:
        success = setup_vector_store()
        sys.exit(0 if success else 1)
    elif args.test:
        success = test_vector_store()
        sys.exit(0 if success else 1)
    elif args.search:
        interactive_search()
    elif args.status:
        show_status()
    else:
        print("Vector Store Runner")
        print("==================")
        print()
        print("Usage:")
        print("  python run_vector_store.py --setup    # Complete setup")
        print("  python run_vector_store.py --test     # Test existing setup")
        print("  python run_vector_store.py --search   # Interactive search")
        print("  python run_vector_store.py --status   # Show system status")
        print()
        print("Quick start: python run_vector_store.py --setup")


if __name__ == "__main__":
    main()
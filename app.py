import json

import pandas as pd
import streamlit as st

from src.config import Config
from src.embedding_generator import EmbeddingGenerator
from src.search_engine import SemanticSearchEngine
from src.transcript_processor import TranscriptProcessor

st.set_page_config(page_title="Semantic Insights PoC", page_icon="🔍", layout="wide")


def normalize_urgency(value):
    """Normalize urgency to an int between 1–5."""
    if value is None:
        return 3

    # Already an int
    if isinstance(value, int):
        return value

    # Try numeric string
    try:
        return int(value)
    except (ValueError, TypeError):
        pass

    # Map descriptive labels
    mapping = {
        "low": 1,
        "medium": 3,
        "high": 5,
        "urgent": 5,
        "critical": 5,
    }
    return mapping.get(str(value).strip().lower(), 3)


def main():
    st.title("🔍 Semantic Insights Search PoC")
    st.markdown("*Demonstrating AI-powered insight extraction and semantic search*")

    # Check API status and show info
    processor = TranscriptProcessor()
    api_status = getattr(processor, "api_available", False)

    if not api_status:
        st.info(
            "Demo Mode: API not available. Using locally stored data as a fallback."
        )

    # Sidebar for data processing
    with st.sidebar:
        st.header("Data Pipeline")

        # Show pipeline steps
        st.markdown("### Pipeline Steps:")
        st.markdown("1. 📄 Raw transcripts")
        st.markdown("2. 🧠 Extract insights (via Gemini)")
        st.markdown("3. 🔢 Generate embeddings (via Gemini)")
        st.markdown("4. 🔍 Semantic search")

        st.divider()

        if st.button(
            "🔄 Process Transcripts", help="Extract insights from raw transcripts"
        ):
            with st.spinner("Extracting insights..."):
                try:
                    insights = processor.process_all_transcripts()
                    if api_status:
                        st.success(
                            f"✅ Processed {len(insights)} transcripts with Gemini API!"
                        )
                    else:
                        st.success(
                            f"✅ Loaded {len(insights)} insights from local file!"
                        )

                    # Show sample result
                    with st.expander("📋 Sample Extracted Insight"):
                        sample = insights[0] if insights else {}
                        st.json(
                            {
                                "participant": sample.get("participant", "N/A"),
                                "primary_goal": sample.get("primary_goal", "N/A"),
                                "main_blocker": sample.get("main_blocker", "N/A"),
                                "business_focus": sample.get("business_focus", "N/A"),
                                "mindset_pattern": sample.get("mindset_pattern", "N/A"),
                            }
                        )

                except Exception as e:
                    st.error(f"❌ Error processing transcripts: {str(e)}")

        if st.button(
            "🧠 Generate Embeddings", help="Create semantic embeddings for search"
        ):
            with st.spinner("Creating embeddings with OpenAI..."):
                try:
                    generator = EmbeddingGenerator()
                    embeddings = generator.process_all_insights()
                    st.success(f"✅ Generated {len(embeddings)} embeddings!")

                    # Show embedding info
                    with st.expander("🔢 Embedding Details"):
                        if embeddings:
                            sample_embedding = embeddings[0]["embedding"]
                            st.write(f"**Dimensions:** {len(sample_embedding)}")
                            st.write(f"**Sample values:** {sample_embedding[:5]}...")
                            st.write(f"**Model:** text-embedding-3-small")

                except Exception as e:
                    st.error(f"❌ Error generating embeddings: {str(e)}")

        st.divider()
        st.markdown("### 💡 Demo Tips")
        st.markdown("- Process transcripts first")
        st.markdown("- Then generate embeddings")
        st.markdown("- Try the example searches")
        st.markdown("- Notice semantic matching!")

    # Main search interface
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("🔍 Semantic Search")

        # Search examples with better descriptions
        example_queries = [
            "Who struggles with delegation and team management?",
            "Find others with pricing and confidence issues",
            "Show me people in e-commerce or SaaS",
            "Who has imposter syndrome or self-doubt?",
            "Find entrepreneurs working too many hours",
            "Who needs help with time management?",
            "Show me people with perfectionist tendencies",
        ]

        selected_example = st.selectbox(
            "🎯 Try an example query:",
            [""] + example_queries,
            help="These examples showcase semantic search capabilities",
        )

        query = st.text_input(
            "🔍 Search query:",
            value=selected_example,
            placeholder="e.g., Who else struggles with time management?",
            help="Ask natural questions about challenges, goals, or business types",
        )

        if query:
            try:
                search_engine = SemanticSearchEngine()
                results = search_engine.search(query, top_k=3)

                if results:
                    st.subheader("🎯 Search Results")
                    st.markdown(f'*Found {len(results)} matches for: "{query}"*')

                    for i, result in enumerate(results, 1):
                        similarity_pct = result["similarity_score"] * 100

                        # Color-code by similarity
                        if similarity_pct > 75:
                            color = "🟢"
                        elif similarity_pct > 60:
                            color = "🟡"
                        else:
                            color = "🔴"

                        with st.expander(
                            f"{color} {i}. {result['participant']} ({similarity_pct:.1f}% match)"
                        ):
                            col_a, col_b = st.columns([1, 1])

                            with col_a:
                                st.markdown("**🎯 Why this matched:**")
                                st.write(result["explanation"])
                                st.markdown("**💼 Business & Stage:**")
                                st.write(
                                    f"• {result['metadata']['business_focus'].title()}"
                                )
                                st.write(
                                    f"• {result['metadata'].get('current_stage', 'Stage unknown')}"
                                )

                            with col_b:
                                st.markdown("**🎯 Goal:**")
                                st.write(result["metadata"]["primary_goal"])
                                st.markdown("**🚫 Main Challenge:**")
                                st.write(result["metadata"]["main_blocker"])
                                st.markdown("**🧠 Mindset Pattern:**")
                                st.write(result["metadata"]["mindset_pattern"])

                else:
                    st.warning(
                        "🤔 No results found. Make sure embeddings are generated first!"
                    )
                    st.info(
                        "💡 Try clicking 'Process Transcripts' then 'Generate Embeddings' in the sidebar."
                    )

            except Exception as e:
                st.error(f"❌ Search error: {str(e)}")
                st.info(
                    "💡 Make sure you've processed transcripts and generated embeddings first."
                )

    with col2:
        st.header("📊 Available Insights")

        try:
            with open(Config.INSIGHTS_PATH, "r") as f:
                insights = json.load(f)

            # Show summary stats
            col_stats1, col_stats2, col_stats3 = st.columns(3)

            with col_stats1:
                st.metric("📝 Total Insights", len(insights))

            with col_stats2:
                business_types = set(
                    insight.get("business_focus", "unknown") for insight in insights
                )
                st.metric("💼 Business Types", len(business_types))

            with col_stats3:
                urgency_values = [
                    normalize_urgency(insight.get("urgency_level", 3))
                    for insight in insights
                ]
                avg_urgency = (
                    sum(urgency_values) / len(urgency_values) if urgency_values else 0
                )
                st.metric("⚡ Avg Urgency", f"{avg_urgency:.1f}/5")

            # Create overview dataframe
            overview_data = []
            for insight in insights:
                overview_data.append(
                    {
                        "👤 Participant": insight["participant"],
                        "💼 Business": insight.get("business_focus", "Unknown").title(),
                        "🎯 Main Goal": (
                            insight.get("primary_goal", "Unknown")[:40] + "..."
                            if len(insight.get("primary_goal", "")) > 40
                            else insight.get("primary_goal", "Unknown")
                        ),
                        "🚫 Top Challenge": (
                            insight.get("main_blocker", "Unknown")[:40] + "..."
                            if len(insight.get("main_blocker", "")) > 40
                            else insight.get("main_blocker", "Unknown")
                        ),
                        "⚡ Urgency": f"{normalize_urgency(insight.get('urgency_level', 3))}/5",
                    }
                )

            df = pd.DataFrame(overview_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Show detailed view option
            if st.checkbox("🔍 Show detailed insights"):
                selected_participant = st.selectbox(
                    "Select participant to view details:",
                    [insight["participant"] for insight in insights],
                )

                selected_insight = next(
                    (
                        insight
                        for insight in insights
                        if insight["participant"] == selected_participant
                    ),
                    None,
                )

                if selected_insight:
                    st.subheader(f"📋 Detailed Profile: {selected_participant}")

                    detail_col1, detail_col2 = st.columns([1, 1])

                    with detail_col1:
                        st.markdown("**🎯 Goals & Challenges:**")
                        st.write(
                            f"**Primary Goal:** {selected_insight.get('primary_goal', 'N/A')}"
                        )
                        st.write(
                            f"**Main Blocker:** {selected_insight.get('main_blocker', 'N/A')}"
                        )

                        secondary = selected_insight.get("secondary_blockers", [])
                        if secondary:
                            st.write("**Other Challenges:**")
                            for blocker in secondary:
                                st.write(f"• {blocker}")

                    with detail_col2:
                        st.markdown("**🧠 Psychology & Context:**")
                        st.write(
                            f"**Mindset Pattern:** {selected_insight.get('mindset_pattern', 'N/A')}"
                        )
                        st.write(
                            f"**Current Stage:** {selected_insight.get('current_stage', 'N/A')}"
                        )
                        st.write(
                            f"**Business Focus:** {selected_insight.get('business_focus', 'N/A')}"
                        )

                        emotions = selected_insight.get("key_emotions", [])
                        if emotions:
                            st.write(f"**Key Emotions:** {', '.join(emotions)}")

        except FileNotFoundError:
            st.info(
                "📝 No insights found. Click 'Process Transcripts' in the sidebar to get started!"
            )

            # Show sample of what will be created
            st.markdown("### 👀 What you'll see after processing:")
            sample_preview = {
                "participant": "Sarah Chen",
                "business_focus": "e-commerce",
                "primary_goal": "Scale from $50K to $200K monthly revenue",
                "main_blocker": "Unable to delegate effectively",
            }
            st.json(sample_preview)


if __name__ == "__main__":
    main()

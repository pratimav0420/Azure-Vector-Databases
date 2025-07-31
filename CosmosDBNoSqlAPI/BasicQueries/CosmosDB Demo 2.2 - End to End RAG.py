"""
Cosmos DB Demo 2.2 - End to End RAG
Equivalent to SQL Demo 2.2 - End to End RAG.sql

This script demonstrates a complete RAG (Retrieval Augmented Generation) system
using Azure Cosmos DB as the vector store and Azure OpenAI for embeddings and completions
"""

from azure.cosmos import CosmosClient, PartitionKey
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import time
import json

# Load environment variables
load_dotenv()

# Configuration
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o")

DATABASE_NAME = "RAGDemoDB"
KNOWLEDGE_CONTAINER = "knowledge_base"
CHAT_CONTAINER = "chat_history"

# Initialize clients
cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION
)

print("🚀 Initialized Azure Cosmos DB and Azure OpenAI clients")
print(f"📚 Embedding model: {EMBEDDING_MODEL}")
print(f"🤖 Generation model: {GENERATION_MODEL}")

# Create database
database = cosmos_client.create_database_if_not_exists(id=DATABASE_NAME)
print(f"✅ Database '{DATABASE_NAME}' ready")

# Vector embedding policy for 1536-dimensional vectors
vector_embedding_policy = {
    "vectorEmbeddings": [
        {
            "path": "/embedding",
            "dataType": "float32",
            "distanceFunction": "cosine",
            "dimensions": 1536
        }
    ]
}

# Indexing policy with vector index
indexing_policy = {
    "indexingMode": "consistent",
    "automatic": True,
    "includedPaths": [{"path": "/*"}],
    "excludedPaths": [{"path": "/embedding/*"}],
    "vectorIndexes": [
        {
            "path": "/embedding",
            "type": "quantizedFlat"
        }
    ]
}

# Create knowledge base container
knowledge_container = database.create_container_if_not_exists(
    id=KNOWLEDGE_CONTAINER,
    partition_key=PartitionKey(path="/category"),
    indexing_policy=indexing_policy,
    vector_embedding_policy=vector_embedding_policy,
    offer_throughput=400
)

# Create chat history container
chat_container = database.create_container_if_not_exists(
    id=CHAT_CONTAINER,
    partition_key=PartitionKey(path="/conversation_id"),
    offer_throughput=400
)

print(f"✅ Containers ready: {KNOWLEDGE_CONTAINER}, {CHAT_CONTAINER}")

def get_embedding(text):
    """Get embedding from Azure OpenAI"""
    try:
        response = openai_client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Error getting embedding: {e}")
        return None

def get_completion(messages, max_tokens=1000):
    """Get completion from Azure OpenAI"""
    try:
        response = openai_client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        return {
            "content": response.choices[0].message.content,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    except Exception as e:
        print(f"❌ Error getting completion: {e}")
        return None

# Step 1: Populate Knowledge Base
print("\n=== Step 1: Building Knowledge Base ===")

# Sample knowledge articles about different topics
knowledge_articles = [
    {
        "title": "Introduction to Machine Learning",
        "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or classifications based on those patterns.",
        "category": "technology",
        "source": "AI Handbook",
        "tags": ["AI", "ML", "algorithms", "data science"]
    },
    {
        "title": "Renewable Energy Sources",
        "content": "Renewable energy comes from natural sources that are constantly replenished. The main types include solar energy from the sun, wind energy from air currents, hydroelectric power from flowing water, and geothermal energy from underground heat. These sources are sustainable and environmentally friendly.",
        "category": "environment",
        "source": "Green Energy Guide",
        "tags": ["solar", "wind", "renewable", "sustainability"]
    },
    {
        "title": "Mediterranean Diet Benefits",
        "content": "The Mediterranean diet is based on traditional eating patterns of countries bordering the Mediterranean Sea. It emphasizes fruits, vegetables, whole grains, legumes, nuts, and olive oil. Research shows it can reduce the risk of heart disease, support brain health, and promote longevity.",
        "category": "health",
        "source": "Nutrition Research",
        "tags": ["diet", "nutrition", "heart health", "Mediterranean"]
    },
    {
        "title": "Deep Learning Neural Networks",
        "content": "Deep learning uses artificial neural networks with multiple layers to model and understand complex patterns in data. These networks can process vast amounts of information and have revolutionized fields like computer vision, natural language processing, and speech recognition.",
        "category": "technology",
        "source": "Deep Learning Textbook",
        "tags": ["deep learning", "neural networks", "AI", "computer vision"]
    },
    {
        "title": "Climate Change Impacts",
        "content": "Climate change refers to long-term shifts in global temperatures and weather patterns. Human activities, particularly fossil fuel burning, have accelerated these changes. Impacts include rising sea levels, extreme weather events, ecosystem disruption, and threats to food security.",
        "category": "environment",
        "source": "Climate Science Report",
        "tags": ["climate change", "global warming", "environment", "sustainability"]
    },
    {
        "title": "Exercise and Mental Health",
        "content": "Regular physical exercise has profound benefits for mental health. It can reduce symptoms of depression and anxiety, improve mood, boost self-esteem, and enhance cognitive function. Exercise releases endorphins and promotes better sleep patterns.",
        "category": "health",
        "source": "Sports Medicine Journal",
        "tags": ["exercise", "mental health", "wellness", "fitness"]
    },
    {
        "title": "Quantum Computing Basics",
        "content": "Quantum computing harnesses quantum mechanical phenomena like superposition and entanglement to process information. Unlike classical bits that are either 0 or 1, quantum bits (qubits) can exist in multiple states simultaneously, potentially solving certain problems exponentially faster.",
        "category": "technology",
        "source": "Quantum Physics Today",
        "tags": ["quantum", "computing", "qubits", "superposition"]
    },
    {
        "title": "Sustainable Agriculture Practices",
        "content": "Sustainable agriculture focuses on producing food while protecting environmental resources for future generations. Key practices include crop rotation, organic farming, integrated pest management, and soil conservation. These methods maintain soil health and biodiversity.",
        "category": "environment",
        "source": "Agricultural Science Review",
        "tags": ["agriculture", "sustainability", "organic farming", "soil health"]
    }
]

# Clear existing knowledge base
print("🗑️ Clearing existing knowledge base...")
try:
    existing_items = knowledge_container.query_items(
        query="SELECT c.id, c.category FROM c WHERE c.document_type = 'knowledge'",
        enable_cross_partition_query=True
    )
    
    for item in existing_items:
        knowledge_container.delete_item(item=item['id'], partition_key=item['category'])
    print("✅ Cleared existing knowledge base")
except Exception as e:
    print(f"Note: {e}")

# Add articles to knowledge base with embeddings
print("📝 Adding articles to knowledge base...")

for i, article in enumerate(knowledge_articles, 1):
    print(f"Processing article {i}/{len(knowledge_articles)}: {article['title']}")
    
    # Create combined text for embedding
    combined_text = f"{article['title']} {article['content']} {' '.join(article['tags'])}"
    
    # Get embedding
    embedding = get_embedding(combined_text)
    
    if embedding:
        # Create document
        doc = {
            "id": f"article_{i}",
            "title": article['title'],
            "content": article['content'],
            "category": article['category'],
            "source": article['source'],
            "tags": article['tags'],
            "combined_text": combined_text,
            "embedding": embedding,
            "embedding_model": EMBEDDING_MODEL,
            "created_at": time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "document_type": "knowledge"
        }
        
        try:
            knowledge_container.create_item(doc)
            print(f"  ✅ Stored with embedding")
        except Exception as e:
            print(f"  ❌ Error storing: {e}")
    else:
        print(f"  ❌ Failed to get embedding")
    
    # Rate limiting
    time.sleep(0.5)

print(f"✅ Knowledge base populated with {len(knowledge_articles)} articles")

# Step 2: Implement RAG Functions
def retrieve_relevant_documents(query, top_k=3):
    """Retrieve relevant documents from knowledge base using vector similarity"""
    print(f"🔍 Searching for: '{query}'")
    
    # Get query embedding
    query_embedding = get_embedding(query)
    if not query_embedding:
        print("❌ Failed to get query embedding")
        return []
    
    # Search for similar documents
    search_query = """
    SELECT TOP @topK
        c.title,
        c.content,
        c.category,
        c.source,
        c.tags,
        1 - VectorDistance('cosine', c.embedding, @queryEmbedding) as relevance_score
    FROM c 
    WHERE c.document_type = 'knowledge'
    ORDER BY VectorDistance('cosine', c.embedding, @queryEmbedding)
    """
    
    results = list(knowledge_container.query_items(
        query=search_query,
        parameters=[
            {"name": "@queryEmbedding", "value": query_embedding},
            {"name": "@topK", "value": top_k}
        ],
        enable_cross_partition_query=True
    ))
    
    print(f"📚 Found {len(results)} relevant documents")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. [{doc['category']}] {doc['title']} (relevance: {doc['relevance_score']:.4f})")
    
    return results

def generate_rag_response(user_query, retrieved_docs, conversation_history=None):
    """Generate response using retrieved documents and optional conversation history"""
    
    # Build context from retrieved documents
    context_parts = []
    for doc in retrieved_docs:
        context_parts.append(f"Title: {doc['title']}\nContent: {doc['content']}\nSource: {doc['source']}\n")
    
    context = "\\n".join(context_parts)
    
    # Build messages
    system_prompt = """You are a knowledgeable AI assistant. Use the provided context documents to answer the user's question accurately and comprehensively. 
    
    Instructions:
    - Base your answer primarily on the provided context
    - If the context doesn't fully answer the question, acknowledge the limitations
    - Cite sources when appropriate
    - Be helpful and conversational
    - If you use information from multiple sources, synthesize it coherently"""
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history if provided
    if conversation_history:
        for msg in conversation_history[-6:]:  # Use last 6 messages for context
            messages.append({"role": "user", "content": msg['user_query']})
            messages.append({"role": "assistant", "content": msg['assistant_response']})
    
    # Add current query with context
    user_prompt = f"""Context Documents:
{context}

User Question: {user_query}

Please provide a comprehensive answer based on the context provided."""
    
    messages.append({"role": "user", "content": user_prompt})
    
    # Get completion
    return get_completion(messages)

def save_conversation(conversation_id, user_query, assistant_response, retrieved_docs, tokens_used):
    """Save conversation to chat history"""
    try:
        message_id = f"{conversation_id}_{int(time.time() * 1000)}"
        
        doc = {
            "id": message_id,
            "conversation_id": conversation_id,
            "user_query": user_query,
            "assistant_response": assistant_response,
            "retrieved_documents": [
                {
                    "title": doc['title'],
                    "category": doc['category'],
                    "relevance_score": doc['relevance_score']
                } for doc in retrieved_docs
            ],
            "tokens_used": tokens_used,
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "document_type": "conversation"
        }
        
        chat_container.create_item(doc)
        return message_id
        
    except Exception as e:
        print(f"❌ Error saving conversation: {e}")
        return None

def get_conversation_history(conversation_id, limit=10):
    """Get conversation history"""
    try:
        history_query = """
        SELECT TOP @limit
            c.user_query,
            c.assistant_response,
            c.timestamp
        FROM c 
        WHERE c.conversation_id = @conversationId 
            AND c.document_type = 'conversation'
        ORDER BY c.timestamp ASC
        """
        
        results = list(chat_container.query_items(
            query=history_query,
            parameters=[
                {"name": "@conversationId", "value": conversation_id},
                {"name": "@limit", "value": limit}
            ],
            enable_cross_partition_query=True
        ))
        
        return results
        
    except Exception as e:
        print(f"❌ Error getting conversation history: {e}")
        return []

# Step 3: RAG Demo Scenarios
print("\n=== Step 3: RAG Demo Scenarios ===")

# Clear previous conversations
print("🗑️ Clearing previous conversations...")
try:
    existing_conversations = chat_container.query_items(
        query="SELECT c.id, c.conversation_id FROM c WHERE c.document_type = 'conversation'",
        enable_cross_partition_query=True
    )
    
    for conv in existing_conversations:
        chat_container.delete_item(item=conv['id'], partition_key=conv['conversation_id'])
    print("✅ Cleared previous conversations")
except Exception as e:
    print(f"Note: {e}")

# Demo 1: Single question RAG
print("\n--- Demo 1: Single Question RAG ---")
conversation_id = "demo_rag_1"
user_query = "What are the benefits of renewable energy for the environment?"

print(f"💭 User: {user_query}")

# Retrieve relevant documents
retrieved_docs = retrieve_relevant_documents(user_query, top_k=3)

if retrieved_docs:
    # Generate response
    print("🤖 Generating response...")
    response = generate_rag_response(user_query, retrieved_docs)
    
    if response:
        print(f"\\n🤖 Assistant: {response['content']}")
        print(f"📊 Tokens used: {response['total_tokens']}")
        
        # Save conversation
        save_conversation(
            conversation_id=conversation_id,
            user_query=user_query,
            assistant_response=response['content'],
            retrieved_docs=retrieved_docs,
            tokens_used=response['total_tokens']
        )

# Demo 2: Follow-up conversation with context
print("\n--- Demo 2: Follow-up with Context ---")

follow_up_query = "How does solar energy specifically work?"
print(f"💭 User: {follow_up_query}")

# Get conversation history
conversation_history = get_conversation_history(conversation_id)
print(f"📚 Using context from {len(conversation_history)} previous messages")

# Retrieve relevant documents
retrieved_docs_2 = retrieve_relevant_documents(follow_up_query, top_k=2)

if retrieved_docs_2:
    # Generate response with conversation history
    response_2 = generate_rag_response(follow_up_query, retrieved_docs_2, conversation_history)
    
    if response_2:
        print(f"\\n🤖 Assistant: {response_2['content']}")
        print(f"📊 Tokens used: {response_2['total_tokens']}")
        
        # Save conversation
        save_conversation(
            conversation_id=conversation_id,
            user_query=follow_up_query,
            assistant_response=response_2['content'],
            retrieved_docs=retrieved_docs_2,
            tokens_used=response_2['total_tokens']
        )

# Demo 3: Multi-domain question
print("\n--- Demo 3: Multi-domain Question ---")
conversation_id_2 = "demo_rag_2"
complex_query = "How can technology help address climate change and improve human health?"

print(f"💭 User: {complex_query}")

# Retrieve relevant documents (higher k for complex question)
retrieved_docs_3 = retrieve_relevant_documents(complex_query, top_k=4)

if retrieved_docs_3:
    response_3 = generate_rag_response(complex_query, retrieved_docs_3)
    
    if response_3:
        print(f"\\n🤖 Assistant: {response_3['content']}")
        print(f"📊 Tokens used: {response_3['total_tokens']}")
        
        save_conversation(
            conversation_id=conversation_id_2,
            user_query=complex_query,
            assistant_response=response_3['content'],
            retrieved_docs=retrieved_docs_3,
            tokens_used=response_3['total_tokens']
        )

# Demo 4: Iterative conversation
print("\n--- Demo 4: Iterative Conversation ---")
conversation_id_3 = "demo_rag_3"

iterative_queries = [
    "Tell me about machine learning",
    "What's the difference between machine learning and deep learning?",
    "How are neural networks used in AI applications?",
    "Can you give me practical examples of where this technology is used today?"
]

for i, query in enumerate(iterative_queries, 1):
    print(f"\\n--- Query {i} ---")
    print(f"💭 User: {query}")
    
    # Get conversation history for context
    history = get_conversation_history(conversation_id_3)
    
    # Retrieve relevant documents
    docs = retrieve_relevant_documents(query, top_k=2)
    
    if docs:
        response = generate_rag_response(query, docs, history)
        
        if response:
            print(f"🤖 Assistant: {response['content'][:300]}..." if len(response['content']) > 300 else f"🤖 Assistant: {response['content']}")
            print(f"📊 Tokens: {response['total_tokens']}")
            
            save_conversation(
                conversation_id=conversation_id_3,
                user_query=query,
                assistant_response=response['content'],
                retrieved_docs=docs,
                tokens_used=response['total_tokens']
            )
    
    time.sleep(1)  # Brief pause between queries

# Analytics and Summary
print("\n=== RAG System Analytics ===")

# Knowledge base stats
kb_stats_query = """
SELECT 
    c.category,
    COUNT(1) as article_count
FROM c 
WHERE c.document_type = 'knowledge'
GROUP BY c.category
ORDER BY article_count DESC
"""

kb_stats = list(knowledge_container.query_items(query=kb_stats_query, enable_cross_partition_query=True))
print("📚 Knowledge Base by Category:")
for stat in kb_stats:
    print(f"  {stat['category']}: {stat['article_count']} articles")

# Conversation stats
conv_stats_query = """
SELECT 
    COUNT(1) as total_conversations,
    SUM(c.tokens_used) as total_tokens,
    AVG(c.tokens_used) as avg_tokens_per_query
FROM c 
WHERE c.document_type = 'conversation'
"""

conv_stats = list(chat_container.query_items(query=conv_stats_query, enable_cross_partition_query=True))
if conv_stats and conv_stats[0]['total_conversations'] > 0:
    stat = conv_stats[0]
    print(f"\\n💬 Conversation Statistics:")
    print(f"  Total queries: {stat['total_conversations']}")
    print(f"  Total tokens used: {stat['total_tokens']}")
    print(f"  Average tokens per query: {stat['avg_tokens_per_query']:.1f}")

# Most retrieved categories
category_retrieval_query = """
SELECT 
    VALUE doc.category
FROM c
JOIN doc IN c.retrieved_documents
WHERE c.document_type = 'conversation'
"""

retrieved_categories = list(chat_container.query_items(query=category_retrieval_query, enable_cross_partition_query=True))
if retrieved_categories:
    from collections import Counter
    category_counts = Counter(retrieved_categories)
    print(f"\\n🎯 Most Retrieved Categories:")
    for category, count in category_counts.most_common():
        print(f"  {category}: {count} times")

print("\\n🎉 End-to-End RAG Demo Completed!")
print("\\n✅ Demonstrated Capabilities:")
print("  • Vector-based document retrieval")
print("  • Context-aware response generation")
print("  • Conversation history management")
print("  • Multi-domain knowledge synthesis")
print("  • Iterative conversation flows")
print("  • Analytics and monitoring")

print(f"\\n🔧 System Configuration:")
print(f"  • Knowledge Base: {len(knowledge_articles)} articles across {len(set(article['category'] for article in knowledge_articles))} categories")
print(f"  • Embedding Model: {EMBEDDING_MODEL} (1536 dimensions)")
print(f"  • Generation Model: {GENERATION_MODEL}")
print(f"  • Vector Index: Quantized Flat (Cosine similarity)")
print(f"  • Storage: Azure Cosmos DB NoSQL API")

# Cleanup (uncomment to delete test data)
print("\\n=== Cleanup ===")
print("To clean up test data, uncomment the cleanup section below")

# # Delete all knowledge articles
# print("Deleting knowledge base...")
# all_articles = knowledge_container.query_items(
#     query="SELECT c.id, c.category FROM c WHERE c.document_type = 'knowledge'",
#     enable_cross_partition_query=True
# )
# 
# for article in all_articles:
#     knowledge_container.delete_item(item=article['id'], partition_key=article['category'])
# 
# # Delete all conversations
# print("Deleting conversations...")
# all_conversations = chat_container.query_items(
#     query="SELECT c.id, c.conversation_id FROM c WHERE c.document_type = 'conversation'",
#     enable_cross_partition_query=True
# )
# 
# for conv in all_conversations:
#     chat_container.delete_item(item=conv['id'], partition_key=conv['conversation_id'])
# 
# # Delete containers
# database.delete_container(knowledge_container)
# database.delete_container(chat_container)
# print("Containers deleted")
# 
# # Delete database  
# cosmos_client.delete_database(database)
# print(f"Database '{DATABASE_NAME}' deleted")

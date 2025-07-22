"""
MongoDB vCore Demo 2.2 - End to End RAG
Equivalent to SQL Demo 2.2 - End to End RAG.sql

This script demonstrates a complete RAG (Retrieval Augmented Generation) system 
using Azure Cosmos DB for MongoDB vCore as the vector store and Azure OpenAI for generation
"""

from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import numpy as np
import time
import json

# Load environment variables
load_dotenv()

# Configuration
MONGODB_CONNECTION_STRING = os.getenv("MONGODB_CONNECTION_STRING")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o")

DATABASE_NAME = "RAGDemoDB"
KNOWLEDGE_COLLECTION = "knowledge_base"
CONVERSATION_COLLECTION = "rag_conversations"

# Validate required environment variables
required_vars = ["MONGODB_CONNECTION_STRING", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize clients
mongo_client = MongoClient(MONGODB_CONNECTION_STRING)
openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION
)

# Get database and collections
database = mongo_client[DATABASE_NAME]
knowledge_collection = database[KNOWLEDGE_COLLECTION]
conversation_collection = database[CONVERSATION_COLLECTION]

print(f"Database '{DATABASE_NAME}' ready")
print(f"Knowledge collection '{KNOWLEDGE_COLLECTION}' ready")
print(f"Conversation collection '{CONVERSATION_COLLECTION}' ready")
print(f"Using embedding model: {EMBEDDING_MODEL}")
print(f"Using generation model: {GENERATION_MODEL}")

# Clear existing data for fresh demo
knowledge_collection.delete_many({})
conversation_collection.delete_many({})
print("Cleared existing RAG data")

# Function to get embedding from Azure OpenAI
def get_embedding(text, model=EMBEDDING_MODEL):
    """Get embedding from Azure OpenAI"""
    try:
        response = openai_client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

# Function to get chat completion from Azure OpenAI
def get_chat_completion(messages, model=GENERATION_MODEL, max_tokens=800):
    """Get chat completion from Azure OpenAI"""
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting chat completion: {e}")
        return None

# Create vector search index for knowledge base
vector_index_model = SearchIndexModel(
    definition={
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 1536,
                "similarity": "cosine"
            }
        ]
    },
    name="knowledge_vector_index"
)

try:
    # Check if index exists
    existing_indexes = list(knowledge_collection.list_search_indexes())
    index_exists = any(idx.get('name') == 'knowledge_vector_index' for idx in existing_indexes)
    
    if not index_exists:
        knowledge_collection.create_search_index(vector_index_model)
        print("✅ Created knowledge base vector search index")
        print("⏳ Waiting for index to be ready...")
        time.sleep(30)  # Wait for index to be built
    else:
        print("✅ Knowledge base vector search index already exists")

except Exception as e:
    print(f"⚠️ Vector index creation: {e}")

# Sample movie knowledge base data
movie_knowledge = [
    {
        "title": "The Shawshank Redemption",
        "content": "The Shawshank Redemption is a 1994 American drama film directed by Frank Darabont. It tells the story of Andy Dufresne, a banker who is sentenced to life in Shawshank State Penitentiary for the murders of his wife and her lover, despite his claims of innocence. The film is based on Stephen King's novella Rita Hayworth and Shawshank Redemption.",
        "genre": "Drama",
        "year": 1994,
        "director": "Frank Darabont",
        "rating": 9.3
    },
    {
        "title": "The Godfather",
        "content": "The Godfather is a 1972 American crime film directed by Francis Ford Coppola. The story follows Vito Corleone, the aging patriarch of an organized crime dynasty, who transfers control of his clandestine empire to his reluctant son Michael. The film features Marlon Brando, Al Pacino, and James Caan.",
        "genre": "Crime",
        "year": 1972,
        "director": "Francis Ford Coppola",
        "rating": 9.2
    },
    {
        "title": "The Dark Knight",
        "content": "The Dark Knight is a 2008 superhero film directed by Christopher Nolan. It is the second installment of Nolan's The Dark Knight Trilogy. Batman faces the Joker, a psychopathic criminal mastermind who wants to plunge Gotham City into anarchy. Heath Ledger's performance as the Joker is widely acclaimed.",
        "genre": "Action/Superhero",
        "year": 2008,
        "director": "Christopher Nolan",
        "rating": 9.0
    },
    {
        "title": "Pulp Fiction",
        "content": "Pulp Fiction is a 1994 American crime film written and directed by Quentin Tarantino. The film is known for its eclectic dialogue, ironic mix of humor and violence, nonlinear narrative structure, and cultural references. It stars John Travolta, Uma Thurman, and Samuel L. Jackson.",
        "genre": "Crime",
        "year": 1994,
        "director": "Quentin Tarantino",
        "rating": 8.9
    },
    {
        "title": "Forrest Gump",
        "content": "Forrest Gump is a 1994 American comedy-drama film directed by Robert Zemeckis. Tom Hanks stars as Forrest Gump, a slow-witted but kind-hearted man from Alabama who witnesses and influences several defining historical events in the 20th century United States.",
        "genre": "Comedy-Drama",
        "year": 1994,
        "director": "Robert Zemeckis",
        "rating": 8.8
    },
    {
        "title": "Inception",
        "content": "Inception is a 2010 science fiction action film written and directed by Christopher Nolan. The film stars Leonardo DiCaprio as Dom Cobb, a professional thief who infiltrates the subconscious of his targets to steal secrets through shared dreaming technology.",
        "genre": "Science Fiction",
        "year": 2010,
        "director": "Christopher Nolan",
        "rating": 8.7
    },
    {
        "title": "The Matrix",
        "content": "The Matrix is a 1999 science fiction action film directed by the Wachowskis. It depicts a dystopian future where humanity is unknowingly trapped inside a simulated reality called the Matrix. Keanu Reeves stars as Neo, a computer hacker who discovers the truth.",
        "genre": "Science Fiction",
        "year": 1999,
        "director": "The Wachowskis",
        "rating": 8.7
    }
]

print(f"\n📚 Building knowledge base with {len(movie_knowledge)} movie entries...")

# Generate embeddings and store knowledge base documents
documents_to_insert = []
successful_embeddings = 0

for movie in movie_knowledge:
    print(f"Processing: {movie['title']}")
    
    # Create combined text for embedding
    combined_text = f"{movie['title']} {movie['content']} Genre: {movie['genre']} Director: {movie['director']} Year: {movie['year']}"
    
    embedding = get_embedding(combined_text)
    
    if embedding:
        document = {
            "title": movie["title"],
            "content": movie["content"],
            "genre": movie["genre"],
            "year": movie["year"],
            "director": movie["director"],
            "rating": movie["rating"],
            "combined_text": combined_text,
            "embedding": embedding,
            "embedding_model": EMBEDDING_MODEL,
            "created_at": time.time()
        }
        documents_to_insert.append(document)
        successful_embeddings += 1
        
        # Rate limiting
        time.sleep(0.5)
    else:
        print(f"⚠️ Failed to generate embedding for: {movie['title']}")

# Bulk insert knowledge base documents
if documents_to_insert:
    try:
        result = knowledge_collection.insert_many(documents_to_insert)
        print(f"✅ Successfully stored {len(result.inserted_ids)} movies in knowledge base")
    except Exception as e:
        print(f"❌ Error during bulk insert: {e}")

print(f"📊 Successfully processed {successful_embeddings} out of {len(movie_knowledge)} movies")

# Function to perform semantic search in knowledge base
def semantic_search_knowledge(query_text, top_k=3, similarity_threshold=0.7):
    """Perform semantic search in the knowledge base"""
    
    # Get embedding for query
    query_embedding = get_embedding(query_text)
    if not query_embedding:
        print(f"Failed to get embedding for query: {query_text}")
        return []
    
    try:
        # Use MongoDB vector search
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "knowledge_vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": top_k * 10,
                    "limit": top_k
                }
            },
            {
                "$project": {
                    "title": 1,
                    "content": 1,
                    "genre": 1,
                    "year": 1,
                    "director": 1,
                    "rating": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        results = list(knowledge_collection.aggregate(pipeline))
        
        # Filter by similarity threshold
        filtered_results = [r for r in results if r.get('score', 0) >= similarity_threshold]
        
        return filtered_results
        
    except Exception as e:
        print(f"Error during vector search: {e}")
        return semantic_search_fallback(query_embedding, top_k, similarity_threshold)

def semantic_search_fallback(query_embedding, top_k=3, similarity_threshold=0.7):
    """Fallback semantic search using manual similarity calculation"""
    
    try:
        # Get all documents with embeddings
        all_docs = list(knowledge_collection.find({"embedding": {"$exists": True}}))
        
        if not all_docs:
            return []
        
        # Calculate similarities
        similarities = []
        query_vector = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vector)
        
        for doc in all_docs:
            if 'embedding' in doc and doc['embedding']:
                doc_vector = np.array(doc['embedding'])
                doc_norm = np.linalg.norm(doc_vector)
                
                if doc_norm > 0 and query_norm > 0:
                    similarity = np.dot(query_vector, doc_vector) / (query_norm * doc_norm)
                    if similarity >= similarity_threshold:
                        doc['score'] = similarity
                        similarities.append(doc)
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x['score'], reverse=True)
        return similarities[:top_k]
        
    except Exception as e:
        print(f"Error in fallback search: {e}")
        return []

# Function to implement complete RAG pipeline
def rag_query(user_question, conversation_id="default"):
    """Complete RAG pipeline: Retrieve relevant knowledge, then generate answer"""
    
    print(f"🔍 RAG Query: '{user_question}'")
    
    # Step 1: Retrieve relevant knowledge using semantic search
    print("📚 Step 1: Retrieving relevant knowledge...")
    relevant_docs = semantic_search_knowledge(user_question, top_k=3)
    
    if not relevant_docs:
        return "I couldn't find relevant information to answer your question."
    
    print(f"✅ Found {len(relevant_docs)} relevant documents")
    
    # Step 2: Format retrieved knowledge as context
    context_parts = []
    for i, doc in enumerate(relevant_docs, 1):
        context_parts.append(
            f"Movie {i}: {doc['title']} ({doc['year']})\n"
            f"Director: {doc['director']}\n"
            f"Genre: {doc['genre']}\n"
            f"Rating: {doc['rating']}/10\n"
            f"Description: {doc['content']}\n"
            f"Relevance Score: {doc.get('score', 'N/A'):.4f}\n"
        )
    
    context = "\n".join(context_parts)
    
    # Step 3: Generate response using GPT-4o with retrieved context
    print("🤖 Step 2: Generating answer with retrieved context...")
    
    system_prompt = """You are a knowledgeable movie expert assistant. Use the provided movie information to answer the user's question accurately and helpfully. 
    
    Guidelines:
    - Base your answer primarily on the provided movie information
    - Be specific about movie titles, years, directors, and other details
    - If the information doesn't fully answer the question, say so
    - Provide engaging and informative responses
    - Reference specific movies from the context when relevant"""
    
    user_prompt = f"""Question: {user_question}
    
    Retrieved Movie Information:
    {context}
    
    Please answer the question based on the provided movie information."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    answer = get_chat_completion(messages)
    
    # Step 4: Store the RAG interaction
    rag_interaction = {
        "conversation_id": conversation_id,
        "timestamp": time.time(),
        "user_question": user_question,
        "retrieved_docs": relevant_docs,
        "context_used": context,
        "generated_answer": answer,
        "embedding_model": EMBEDDING_MODEL,
        "generation_model": GENERATION_MODEL
    }
    
    try:
        conversation_collection.insert_one(rag_interaction)
        print("✅ Stored RAG interaction")
    except Exception as e:
        print(f"⚠️ Could not store RAG interaction: {e}")
    
    return {
        "question": user_question,
        "answer": answer,
        "sources": relevant_docs,
        "context": context
    }

# Demo: RAG Queries
print("\n🎯 **RAG Demo: Movie Question Answering**\n")

# RAG Query 1: General movie recommendation
print("="*80)
query1 = "What are some great crime movies and why are they considered classics?"
result1 = rag_query(query1, "demo_conversation_1")

if result1:
    print(f"\n**Question:** {result1['question']}")
    print(f"\n**Answer:** {result1['answer']}")
    print(f"\n**Sources Used:** {len(result1['sources'])} movies")
    for source in result1['sources']:
        print(f"  • {source['title']} ({source['year']}) - Score: {source.get('score', 'N/A'):.4f}")

# RAG Query 2: Specific director inquiry
print("\n" + "="*80)
query2 = "Tell me about Christopher Nolan's movies and their unique characteristics."
result2 = rag_query(query2, "demo_conversation_1")

if result2:
    print(f"\n**Question:** {result2['question']}")
    print(f"\n**Answer:** {result2['answer']}")
    print(f"\n**Sources Used:** {len(result2['sources'])} movies")

# RAG Query 3: Genre-specific question
print("\n" + "="*80)
query3 = "What makes a good science fiction movie? Give me examples."
result3 = rag_query(query3, "demo_conversation_2")

if result3:
    print(f"\n**Question:** {result3['question']}")
    print(f"\n**Answer:** {result3['answer']}")
    print(f"\n**Sources Used:** {len(result3['sources'])} movies")

# RAG Query 4: Year-based inquiry
print("\n" + "="*80)
query4 = "What were some notable movies from 1994 and what made them special?"
result4 = rag_query(query4, "demo_conversation_2")

if result4:
    print(f"\n**Question:** {result4['question']}")
    print(f"\n**Answer:** {result4['answer']}")
    print(f"\n**Sources Used:** {len(result4['sources'])} movies")

# Demo: RAG Analytics
print("\n📊 **RAG System Analytics**")

# Knowledge base statistics
total_knowledge = knowledge_collection.count_documents({})
print(f"Knowledge base size: {total_knowledge} documents")

# Conversation statistics
total_interactions = conversation_collection.count_documents({})
unique_conversations = len(conversation_collection.distinct("conversation_id"))
print(f"Total RAG interactions: {total_interactions}")
print(f"Unique conversations: {unique_conversations}")

# Most common question topics (simple keyword analysis)
print("\nQuestion analysis:")
pipeline = [
    {"$project": {"user_question": 1, "words": {"$split": ["$user_question", " "]}}},
    {"$unwind": "$words"},
    {"$match": {"words": {"$regex": "^[A-Za-z]{4,}$"}}},  # Words with 4+ letters
    {"$group": {"_id": {"$toLower": "$words"}, "count": {"$sum": 1}}},
    {"$sort": {"count": -1}},
    {"$limit": 10}
]

try:
    word_counts = list(conversation_collection.aggregate(pipeline))
    print("Most common question words:")
    for word_info in word_counts:
        print(f"  {word_info['_id']}: {word_info['count']}")
except Exception as e:
    print(f"Could not analyze questions: {e}")

# Average relevance scores
print("\nRetrieval quality:")
pipeline = [
    {"$unwind": "$retrieved_docs"},
    {"$group": {
        "_id": None,
        "avg_score": {"$avg": "$retrieved_docs.score"},
        "min_score": {"$min": "$retrieved_docs.score"},
        "max_score": {"$max": "$retrieved_docs.score"}
    }}
]

try:
    score_stats = list(conversation_collection.aggregate(pipeline))
    if score_stats:
        stats = score_stats[0]
        print(f"Average relevance score: {stats.get('avg_score', 0):.4f}")
        print(f"Score range: {stats.get('min_score', 0):.4f} - {stats.get('max_score', 0):.4f}")
except Exception as e:
    print(f"Could not calculate score statistics: {e}")

print("\n✅ End-to-End RAG demo completed!")
print("This demo showed:")
print("• Complete RAG pipeline with MongoDB vCore as vector store")
print("• Azure OpenAI embedding generation for knowledge base")
print("• Semantic search and retrieval of relevant documents")
print("• GPT-4o answer generation using retrieved context")
print("• RAG interaction storage and analytics")
print("• Multi-turn RAG conversations")
print("• Knowledge base management and vector indexing")
print("• RAG system performance monitoring")

# Cleanup (uncomment to clean up test data)
# knowledge_collection.delete_many({})
# conversation_collection.delete_many({})
# print("🧹 Cleaned up RAG data")

# Close connection
mongo_client.close()

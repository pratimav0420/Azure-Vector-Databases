"""
MongoDB vCore Demo 1.2 - Simple Embedding
Equivalent to SQL Demo 1.2 - Simple Embedding.sql

This script demonstrates embedding generation and storage in Azure Cosmos DB for MongoDB vCore
"""

from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import numpy as np
import time

# Load environment variables
load_dotenv()

# Configuration
MONGODB_CONNECTION_STRING = os.getenv("MONGODB_CONNECTION_STRING")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

DATABASE_NAME = "EmbeddingTestDB"
COLLECTION_NAME = "EmbeddingItems"

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

# Get database and collection
database = mongo_client[DATABASE_NAME]
collection = database[COLLECTION_NAME]

print(f"Database '{DATABASE_NAME}' ready")
print(f"Collection '{COLLECTION_NAME}' ready")
print(f"Using embedding model: {EMBEDDING_MODEL}")

# Clear existing data for fresh demo
collection.delete_many({})
print("Cleared existing test data")

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
        print(f"Error getting embedding for '{text}': {e}")
        return None

# Create vector search index for 1536-dimensional embeddings
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
    name="embedding_vector_index"
)

try:
    # Check if index exists
    existing_indexes = list(collection.list_search_indexes())
    index_exists = any(idx.get('name') == 'embedding_vector_index' for idx in existing_indexes)
    
    if not index_exists:
        collection.create_search_index(vector_index_model)
        print("✅ Created embedding vector search index")
        print("⏳ Waiting for index to be ready...")
        time.sleep(30)  # Wait for index to be built
    else:
        print("✅ Embedding vector search index already exists")

except Exception as e:
    print(f"⚠️ Vector index creation: {e}")

# Test embedding generation for a simple word
print("\n🧪 Testing embedding generation:")
test_text = "Apple"
print(f"Generating embedding for: '{test_text}'")

apple_embedding = get_embedding(test_text)
if apple_embedding:
    print(f"✅ Successfully generated embedding")
    print(f"Embedding dimensions: {len(apple_embedding)}")
    print(f"First 10 dimensions: {apple_embedding[:10]}")
    
    # Store in MongoDB
    apple_doc = {
        "text": test_text,
        "embedding": apple_embedding,
        "created_at": time.time()
    }
    
    try:
        result = collection.insert_one(apple_doc)
        print(f"✅ Stored embedding in MongoDB with ID: {result.inserted_id}")
    except Exception as e:
        print(f"❌ Error storing embedding: {e}")
else:
    print("❌ Failed to generate embedding")

# Generate embeddings for multiple test words
test_words = ["Apple", "Banana", "Orange", "Dog", "Cat", "Bird", "Car", "Bicycle", "Train"]

print(f"\n📝 Generating embeddings for {len(test_words)} words...")

documents_to_insert = []
successful_embeddings = 0

for word in test_words:
    print(f"Processing: {word}")
    embedding = get_embedding(word)
    
    if embedding:
        document = {
            "text": word,
            "embedding": embedding,
            "embedding_model": EMBEDDING_MODEL,
            "created_at": time.time()
        }
        documents_to_insert.append(document)
        successful_embeddings += 1
        
        # Rate limiting
        time.sleep(0.5)
    else:
        print(f"⚠️ Failed to generate embedding for: {word}")

# Bulk insert all documents
if documents_to_insert:
    try:
        result = collection.insert_many(documents_to_insert)
        print(f"✅ Successfully stored {len(result.inserted_ids)} embeddings in MongoDB")
    except Exception as e:
        print(f"❌ Error during bulk insert: {e}")

print(f"📊 Successfully generated {successful_embeddings} out of {len(test_words)} embeddings")

# Function to perform semantic similarity search
def semantic_search(collection, query_text, top_k=5):
    """Perform semantic search using Azure OpenAI embeddings"""
    
    # Get embedding for query text
    query_embedding = get_embedding(query_text)
    if not query_embedding:
        print(f"Failed to get embedding for query: {query_text}")
        return []
    
    try:
        # Use MongoDB vector search
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "embedding_vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": top_k * 5,
                    "limit": top_k
                }
            },
            {
                "$project": {
                    "text": 1,
                    "embedding_model": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        results = list(collection.aggregate(pipeline))
        return results
        
    except Exception as e:
        print(f"Error during vector search: {e}")
        
        # Fallback: manual similarity calculation
        print("🔄 Using fallback similarity calculation...")
        return semantic_search_fallback(collection, query_embedding, top_k)

def semantic_search_fallback(collection, query_embedding, top_k=5):
    """Fallback semantic search using manual cosine similarity"""
    
    try:
        # Get all documents with embeddings
        all_docs = list(collection.find({"embedding": {"$exists": True}}))
        
        if not all_docs:
            print("No documents with embeddings found")
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
                    doc['score'] = similarity
                    similarities.append(doc)
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x['score'], reverse=True)
        return similarities[:top_k]
        
    except Exception as e:
        print(f"Error in fallback search: {e}")
        return []

# Test semantic similarity searches
print("\n🔍 Semantic Similarity Search Tests:")

# Test 1: Search for fruits
query1 = "fruit"
print(f"\n1️⃣ Query: '{query1}'")
results1 = semantic_search(collection, query1, top_k=3)
for i, result in enumerate(results1, 1):
    score = result.get('score', 0)
    print(f"   {i}. {result['text']} (score: {score:.4f})")

# Test 2: Search for animals
query2 = "animal"
print(f"\n2️⃣ Query: '{query2}'")
results2 = semantic_search(collection, query2, top_k=3)
for i, result in enumerate(results2, 1):
    score = result.get('score', 0)
    print(f"   {i}. {result['text']} (score: {score:.4f})")

# Test 3: Search for transportation
query3 = "transportation vehicle"
print(f"\n3️⃣ Query: '{query3}'")
results3 = semantic_search(collection, query3, top_k=3)
for i, result in enumerate(results3, 1):
    score = result.get('score', 0)
    print(f"   {i}. {result['text']} (score: {score:.4f})")

# Test 4: Find similar to a specific word
query4 = "Apple"
print(f"\n4️⃣ Find words similar to: '{query4}'")
results4 = semantic_search(collection, query4, top_k=5)
for i, result in enumerate(results4, 1):
    score = result.get('score', 0)
    # Skip the exact match for this demo
    if result['text'].lower() != query4.lower():
        print(f"   {i}. {result['text']} (score: {score:.4f})")

# Demonstrate finding most and least similar pairs
print("\n🔗 Similarity Analysis:")

# Get embeddings for comparison
apple_doc = collection.find_one({"text": "Apple"})
dog_doc = collection.find_one({"text": "Dog"})
banana_doc = collection.find_one({"text": "Banana"})
cat_doc = collection.find_one({"text": "Cat"})

if apple_doc and banana_doc and apple_doc.get('embedding') and banana_doc.get('embedding'):
    apple_vec = np.array(apple_doc['embedding'])
    banana_vec = np.array(banana_doc['embedding'])
    similarity = np.dot(apple_vec, banana_vec) / (np.linalg.norm(apple_vec) * np.linalg.norm(banana_vec))
    print(f"Apple vs Banana similarity: {similarity:.4f}")

if apple_doc and dog_doc and apple_doc.get('embedding') and dog_doc.get('embedding'):
    apple_vec = np.array(apple_doc['embedding'])
    dog_vec = np.array(dog_doc['embedding'])
    similarity = np.dot(apple_vec, dog_vec) / (np.linalg.norm(apple_vec) * np.linalg.norm(dog_vec))
    print(f"Apple vs Dog similarity: {similarity:.4f}")

if dog_doc and cat_doc and dog_doc.get('embedding') and cat_doc.get('embedding'):
    dog_vec = np.array(dog_doc['embedding'])
    cat_vec = np.array(cat_doc['embedding'])
    similarity = np.dot(dog_vec, cat_vec) / (np.linalg.norm(dog_vec) * np.linalg.norm(cat_vec))
    print(f"Dog vs Cat similarity: {similarity:.4f}")

# Collection statistics
print("\n📊 Collection Statistics:")
total_docs = collection.count_documents({})
docs_with_embeddings = collection.count_documents({"embedding": {"$exists": True}})
print(f"Total documents: {total_docs}")
print(f"Documents with embeddings: {docs_with_embeddings}")

# Check embedding model consistency
models_used = collection.distinct("embedding_model")
print(f"Embedding models used: {models_used}")

# Index information
print("\n🔍 Search Index Information:")
try:
    indexes = list(collection.list_search_indexes())
    for idx in indexes:
        print(f"Index name: {idx.get('name')}")
        print(f"Status: {idx.get('status')}")
        if 'definition' in idx:
            fields = idx['definition'].get('fields', [])
            for field in fields:
                if field.get('type') == 'vector':
                    print(f"Vector path: {field.get('path')}")
                    print(f"Dimensions: {field.get('numDimensions')}")
                    print(f"Similarity: {field.get('similarity')}")
except Exception as e:
    print(f"Could not retrieve index information: {e}")

print("\n✅ Simple embedding demo completed!")
print("This demo showed:")
print("• Azure OpenAI embedding generation using text-embedding-ada-002")
print("• Storing 1536-dimensional embeddings in MongoDB documents")
print("• Vector similarity search using MongoDB $vectorSearch")
print("• Semantic search capabilities with natural language queries")
print("• Cosine similarity calculations for embedding comparison")
print("• Fallback similarity search when vector search is unavailable")

# Cleanup (uncomment to clean up test data)
# collection.delete_many({})
# print("🧹 Cleaned up test data")

# Close connections
mongo_client.close()

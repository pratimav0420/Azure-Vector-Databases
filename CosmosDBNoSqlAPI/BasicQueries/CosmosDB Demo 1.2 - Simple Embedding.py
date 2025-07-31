"""
Cosmos DB Demo 1.2 - Simple Embedding
Equivalent to SQL Demo 1.2 - Simple Embedding.sql

This script demonstrates embedding generation and storage in Azure Cosmos DB NoSQL API
"""

from azure.cosmos import CosmosClient, PartitionKey
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Configuration
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

DATABASE_NAME = "EmbeddingTestDB"
CONTAINER_NAME = "TextEmbeddings"

# Initialize clients
cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION
)

print("Initialized Azure Cosmos DB and Azure OpenAI clients")

# Create database
database = cosmos_client.create_database_if_not_exists(id=DATABASE_NAME)
print(f"Database '{DATABASE_NAME}' ready")

# Define vector embedding policy for 1536-dimensional vectors (text-embedding-ada-002)
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

# Define indexing policy with vector index
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

# Create container
container = database.create_container_if_not_exists(
    id=CONTAINER_NAME,
    partition_key=PartitionKey(path="/text_category"),
    indexing_policy=indexing_policy,
    vector_embedding_policy=vector_embedding_policy,
    offer_throughput=400
)
print(f"Container '{CONTAINER_NAME}' ready with vector capabilities")

def get_embedding(text):
    """Get embedding from Azure OpenAI"""
    try:
        response = openai_client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding for '{text}': {e}")
        return None

# Test texts for embedding
test_texts = [
    {"text": "The quick brown fox jumps over the lazy dog", "category": "animals"},
    {"text": "Machine learning and artificial intelligence", "category": "technology"},
    {"text": "Cooking pasta with tomato sauce", "category": "food"},
    {"text": "The cat sits on the mat", "category": "animals"},
    {"text": "Deep neural networks and transformers", "category": "technology"},
    {"text": "Baking bread in the oven", "category": "food"},
    {"text": "A beautiful sunset over the ocean", "category": "nature"},
    {"text": "Mountain hiking in the Alps", "category": "nature"}
]

print("\n=== Generating Embeddings ===")

# Generate embeddings and store in Cosmos DB
stored_items = []
for i, item in enumerate(test_texts, 1):
    print(f"Processing {i}/{len(test_texts)}: {item['text'][:50]}...")
    
    # Get embedding
    embedding = get_embedding(item['text'])
    
    if embedding:
        # Create document
        doc = {
            "id": str(i),
            "text": item['text'],
            "text_category": item['category'],
            "embedding": embedding,
            "embedding_model": EMBEDDING_MODEL,
            "created_at": time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "document_type": "text_embedding"
        }
        
        try:
            container.create_item(doc)
            stored_items.append(doc)
            print(f"  ✅ Stored with embedding ({len(embedding)} dimensions)")
        except Exception as e:
            print(f"  ❌ Error storing: {e}")
    else:
        print(f"  ❌ Failed to get embedding")
    
    # Rate limiting
    time.sleep(0.5)

print(f"\n=== Stored {len(stored_items)} items with embeddings ===")

# Query all stored items
print("\n=== All Stored Texts ===")
all_texts_query = "SELECT c.id, c.text, c.text_category, ARRAY_LENGTH(c.embedding) as embedding_dims FROM c WHERE c.document_type = 'text_embedding' ORDER BY c.id"
all_texts = list(container.query_items(query=all_texts_query, enable_cross_partition_query=True))

for item in all_texts:
    print(f"ID: {item['id']}, Category: {item['text_category']}, Dims: {item['embedding_dims']}")
    print(f"  Text: {item['text']}")
    print()

# Semantic search example
print("\n=== Semantic Search Examples ===")

def semantic_search(query_text, top_k=3):
    """Perform semantic search using embeddings"""
    print(f"\nSearching for: '{query_text}'")
    
    # Get query embedding
    query_embedding = get_embedding(query_text)
    if not query_embedding:
        print("Failed to get query embedding")
        return
    
    # Search for similar texts
    search_query = """
    SELECT TOP @topK
        c.text,
        c.text_category,
        1 - VectorDistance('cosine', c.embedding, @queryEmbedding) as similarity_score
    FROM c 
    WHERE c.document_type = 'text_embedding'
    ORDER BY VectorDistance('cosine', c.embedding, @queryEmbedding)
    """
    
    results = list(container.query_items(
        query=search_query,
        parameters=[
            {"name": "@queryEmbedding", "value": query_embedding},
            {"name": "@topK", "value": top_k}
        ],
        enable_cross_partition_query=True
    ))
    
    print("Most similar texts:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. [{result['text_category']}] {result['text']} (similarity: {result['similarity_score']:.4f})")

# Example searches
semantic_search("puppies and dogs playing", top_k=3)
semantic_search("artificial intelligence and neural networks", top_k=3)
semantic_search("cooking delicious food", top_k=3)
semantic_search("beautiful landscapes and scenery", top_k=3)

# Category-based similarity analysis
print("\n=== Category Similarity Analysis ===")

categories = ["animals", "technology", "food", "nature"]
for category in categories:
    print(f"\n--- {category.upper()} CATEGORY ---")
    
    category_query = """
    SELECT 
        c.text,
        c.text_category
    FROM c 
    WHERE c.document_type = 'text_embedding' AND c.text_category = @category
    ORDER BY c.id
    """
    
    category_items = list(container.query_items(
        query=category_query,
        parameters=[{"name": "@category", "value": category}],
        enable_cross_partition_query=True
    ))
    
    for item in category_items:
        print(f"  • {item['text']}")

# Cross-category similarity
print("\n=== Cross-Category Similarity ===")
print("Finding technology texts most similar to animal texts...")

cross_similarity_query = """
SELECT 
    t.text as tech_text,
    a.text as animal_text,
    1 - VectorDistance('cosine', t.embedding, a.embedding) as similarity_score
FROM c as t
JOIN c as a
WHERE t.document_type = 'text_embedding' 
    AND a.document_type = 'text_embedding'
    AND t.text_category = 'technology' 
    AND a.text_category = 'animals'
ORDER BY similarity_score DESC
"""

cross_results = list(container.query_items(query=cross_similarity_query, enable_cross_partition_query=True))

print("Technology-Animal similarities:")
for result in cross_results:
    print(f"  Tech: '{result['tech_text']}'")
    print(f"  Animal: '{result['animal_text']}'")
    print(f"  Similarity: {result['similarity_score']:.4f}")
    print()

# Performance test
print("\n=== Performance Test ===")
start_time = time.time()

perf_query = """
SELECT TOP 5
    c.text,
    c.text_category,
    VectorDistance('cosine', c.embedding, @testVector) as distance
FROM c 
WHERE c.document_type = 'text_embedding'
ORDER BY VectorDistance('cosine', c.embedding, @testVector)
"""

# Use the first stored embedding as test vector
if stored_items:
    test_vector = stored_items[0]['embedding']
    
    perf_results = list(container.query_items(
        query=perf_query,
        parameters=[{"name": "@testVector", "value": test_vector}],
        enable_cross_partition_query=True
    ))
    
    end_time = time.time()
    print(f"Vector similarity query executed in {(end_time - start_time)*1000:.2f} ms")
    print(f"Returned {len(perf_results)} results")

# Cleanup (uncomment to delete test data)
print("\n=== Cleanup ===")
print("To clean up test data, uncomment the cleanup section below")

# # Delete all test items
# print("Deleting test data...")
# all_items = container.query_items(
#     query="SELECT c.id, c.text_category FROM c WHERE c.document_type = 'text_embedding'",
#     enable_cross_partition_query=True
# )
# 
# for item in all_items:
#     try:
#         container.delete_item(item=item['id'], partition_key=item['text_category'])
#         print(f"Deleted item {item['id']}")
#     except Exception as e:
#         print(f"Error deleting item {item['id']}: {e}")
# 
# # Delete container
# database.delete_container(container)
# print(f"Container '{CONTAINER_NAME}' deleted")
# 
# # Delete database  
# cosmos_client.delete_database(database)
# print(f"Database '{DATABASE_NAME}' deleted")

print("\nEmbedding demo completed!")
print(f"Used embedding model: {EMBEDDING_MODEL}")
print(f"Vector dimensions: 1536")
print("Demonstrates: embedding generation, storage, and semantic search with Cosmos DB")

"""
MongoDB vCore Demo 1.1 - Simple Vector Query
Equivalent to SQL Demo 1.1 - Simple Vector Query.sql

This script demonstrates basic vector operations in Azure Cosmos DB for MongoDB vCore
"""

from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
import os
from dotenv import load_dotenv
import numpy as np
import time

# Load environment variables
load_dotenv()

# Configuration
MONGODB_CONNECTION_STRING = os.getenv("MONGODB_CONNECTION_STRING")
DATABASE_NAME = "VectorTestDB"
COLLECTION_NAME = "TestItems"

# Initialize client
client = MongoClient(MONGODB_CONNECTION_STRING)

# Get database and collection
database = client[DATABASE_NAME]
collection = database[COLLECTION_NAME]

print(f"Database '{DATABASE_NAME}' ready")
print(f"Collection '{COLLECTION_NAME}' ready")

# Clear existing data for fresh demo
collection.delete_many({})
print("Cleared existing test data")

# Create vector search index for 2-dimensional vectors (for simple demo)
vector_index_model = SearchIndexModel(
    definition={
        "fields": [
            {
                "type": "vector",
                "path": "vectorValue",
                "numDimensions": 2,
                "similarity": "cosine"
            }
        ]
    },
    name="simple_vector_index"
)

try:
    # Check if index exists
    existing_indexes = list(collection.list_search_indexes())
    index_exists = any(idx.get('name') == 'simple_vector_index' for idx in existing_indexes)
    
    if not index_exists:
        collection.create_search_index(vector_index_model)
        print("✅ Created simple vector search index")
        print("⏳ Waiting for index to be ready...")
        time.sleep(20)  # Wait for index to be built
    else:
        print("✅ Vector search index already exists")

except Exception as e:
    print(f"⚠️ Vector index creation: {e}")

# Insert test data with 2D vectors
test_data = [
    {"stringValue": "Apple", "vectorValue": [10, 50]},
    {"stringValue": "Banana", "vectorValue": [12, 48]},
    {"stringValue": "Dog", "vectorValue": [48, 12]},
    {"stringValue": "Cat", "vectorValue": [50, 10]},
    {"stringValue": "Cilantro", "vectorValue": [10, 87]},
    {"stringValue": "Coriander", "vectorValue": [10, 87]}
]

try:
    result = collection.insert_many(test_data)
    print(f"✅ Inserted {len(result.inserted_ids)} test documents")
except Exception as e:
    print(f"❌ Error inserting test data: {e}")

# Display all test data
print("\n📋 All test data:")
for doc in collection.find({}, {"_id": 0}):
    print(f"String: {doc['stringValue']}, Vector: {doc['vectorValue']}")

# Function to calculate cosine similarity manually (fallback)
def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    return dot_product / (norm1 * norm2)

# Function to perform vector similarity search
def vector_similarity_search(collection, query_vector, top_k=5):
    """Perform vector similarity search using MongoDB vector search or fallback"""
    
    try:
        # Try MongoDB vector search first
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "simple_vector_index",
                    "path": "vectorValue",
                    "queryVector": query_vector,
                    "numCandidates": 20,
                    "limit": top_k
                }
            },
            {
                "$project": {
                    "stringValue": 1,
                    "vectorValue": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        results = list(collection.aggregate(pipeline))
        
        if results:
            print("✅ Using MongoDB vector search")
            return results
        else:
            raise Exception("No results from vector search")
            
    except Exception as e:
        print(f"⚠️ Vector search failed: {e}")
        print("🔄 Using fallback similarity calculation...")
        
        # Fallback: manual similarity calculation
        all_docs = list(collection.find({}, {"stringValue": 1, "vectorValue": 1}))
        similarities = []
        
        for doc in all_docs:
            similarity = cosine_similarity(query_vector, doc['vectorValue'])
            doc['score'] = similarity
            similarities.append(doc)
        
        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x['score'], reverse=True)
        return similarities[:top_k]

# Test similarity calculations between specific items
print("\n🔍 Similarity Calculations:")

# Dog vs Cat similarity
dog_doc = collection.find_one({"stringValue": "Dog"})
cat_doc = collection.find_one({"stringValue": "Cat"})

if dog_doc and cat_doc:
    dog_vector = dog_doc['vectorValue']
    cat_vector = cat_doc['vectorValue']
    similarity = cosine_similarity(dog_vector, cat_vector)
    print(f"Dog vs Cat similarity: {similarity:.4f}")

# Apple vs Banana similarity
apple_doc = collection.find_one({"stringValue": "Apple"})
banana_doc = collection.find_one({"stringValue": "Banana"})

if apple_doc and banana_doc:
    apple_vector = apple_doc['vectorValue']
    banana_vector = banana_doc['vectorValue']
    similarity = cosine_similarity(apple_vector, banana_vector)
    print(f"Apple vs Banana similarity: {similarity:.4f}")

# Cilantro vs Coriander similarity (should be very high - identical vectors)
cilantro_doc = collection.find_one({"stringValue": "Cilantro"})
coriander_doc = collection.find_one({"stringValue": "Coriander"})

if cilantro_doc and coriander_doc:
    cilantro_vector = cilantro_doc['vectorValue']
    coriander_vector = coriander_doc['vectorValue']
    similarity = cosine_similarity(cilantro_vector, coriander_vector)
    print(f"Cilantro vs Coriander similarity: {similarity:.4f}")

# Find items most similar to Coriander
print("\n🎯 Items most similar to 'Coriander':")
if coriander_doc:
    coriander_vector = coriander_doc['vectorValue']
    similar_items = vector_similarity_search(collection, coriander_vector, top_k=6)
    
    for i, item in enumerate(similar_items, 1):
        score = item.get('score', 0)
        print(f"{i}. {item['stringValue']} (similarity: {score:.4f})")

# Test vector search with different query vectors
print("\n🧪 Vector Search Tests:")

# Test 1: Query with a fruit-like vector
fruit_query = [11, 49]  # Similar to Apple and Banana
print(f"\nQuery vector {fruit_query} (fruit-like):")
results = vector_similarity_search(collection, fruit_query, top_k=3)
for i, item in enumerate(results, 1):
    score = item.get('score', 0)
    print(f"  {i}. {item['stringValue']} (similarity: {score:.4f})")

# Test 2: Query with an animal-like vector
animal_query = [49, 11]  # Similar to Dog and Cat
print(f"\nQuery vector {animal_query} (animal-like):")
results = vector_similarity_search(collection, animal_query, top_k=3)
for i, item in enumerate(results, 1):
    score = item.get('score', 0)
    print(f"  {i}. {item['stringValue']} (similarity: {score:.4f})")

# Test 3: Query with an herb-like vector
herb_query = [10, 85]  # Similar to Cilantro and Coriander
print(f"\nQuery vector {herb_query} (herb-like):")
results = vector_similarity_search(collection, herb_query, top_k=3)
for i, item in enumerate(results, 1):
    score = item.get('score', 0)
    print(f"  {i}. {item['stringValue']} (similarity: {score:.4f})")

# Collection statistics
print("\n📊 Collection Statistics:")
total_docs = collection.count_documents({})
print(f"Total documents: {total_docs}")

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

print("\n✅ Simple vector query demo completed!")
print("This demo showed:")
print("• Basic 2D vector storage in MongoDB documents")
print("• Vector similarity search using MongoDB $vectorSearch")
print("• Cosine similarity calculations")
print("• Fallback similarity search when vector search is unavailable")
print("• Vector search index creation and management")

# Cleanup (uncomment to clean up test data)
# collection.delete_many({})
# print("🧹 Cleaned up test data")

# Close connection
client.close()

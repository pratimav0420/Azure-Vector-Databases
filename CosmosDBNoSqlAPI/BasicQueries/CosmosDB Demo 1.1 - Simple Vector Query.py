"""
Cosmos DB Demo 1.1 - Simple Vector Query
Equivalent to SQL Demo 1.1 - Simple Vector Query.sql

This script demonstrates basic vector operations in Azure Cosmos DB NoSQL API
"""

from azure.cosmos import CosmosClient, PartitionKey
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")
DATABASE_NAME = "VectorTestDB"
CONTAINER_NAME = "TestItems"

# Initialize client
client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)

# Create database
database = client.create_database_if_not_exists(id=DATABASE_NAME)
print(f"Database '{DATABASE_NAME}' ready")

# Define vector embedding policy for 2-dimensional vectors (for simple demo)
vector_embedding_policy = {
    "vectorEmbeddings": [
        {
            "path": "/vectorValue",
            "dataType": "float32",
            "distanceFunction": "cosine",
            "dimensions": 2
        }
    ]
}

# Define indexing policy with vector index
indexing_policy = {
    "indexingMode": "consistent",
    "automatic": True,
    "includedPaths": [{"path": "/*"}],
    "excludedPaths": [{"path": "/vectorValue/*"}],
    "vectorIndexes": [
        {
            "path": "/vectorValue",
            "type": "quantizedFlat"
        }
    ]
}

# Create container
container = database.create_container_if_not_exists(
    id=CONTAINER_NAME,
    partition_key=PartitionKey(path="/stringValue"),
    indexing_policy=indexing_policy,
    vector_embedding_policy=vector_embedding_policy,
    offer_throughput=400
)
print(f"Container '{CONTAINER_NAME}' ready with vector capabilities")

# Insert test data
test_items = [
    {
        "id": "1",
        "stringValue": "Apple",
        "vectorValue": [10.0, 50.0]
    },
    {
        "id": "2", 
        "stringValue": "Banana",
        "vectorValue": [12.0, 48.0]
    },
    {
        "id": "3",
        "stringValue": "Dog", 
        "vectorValue": [48.0, 12.0]
    },
    {
        "id": "4",
        "stringValue": "Cat",
        "vectorValue": [50.0, 10.0]
    },
    {
        "id": "5",
        "stringValue": "Cilantro",
        "vectorValue": [10.0, 87.0]
    },
    {
        "id": "6",
        "stringValue": "Coriander", 
        "vectorValue": [10.0, 87.0]
    }
]

print("\nInserting test data...")
for item in test_items:
    try:
        container.create_item(item)
        print(f"Inserted: {item['stringValue']}")
    except Exception as e:
        print(f"Error inserting {item['stringValue']}: {e}")

# Query all items
print("\n=== All Items ===")
all_items_query = "SELECT c.stringValue, c.vectorValue FROM c ORDER BY c.stringValue"
all_items = list(container.query_items(query=all_items_query, enable_cross_partition_query=True))

for item in all_items:
    print(f"{item['stringValue']}: {item['vectorValue']}")

# Calculate similarity between Dog and Cat
print("\n=== Similarity: Dog vs Cat ===")
dog_cat_query = """
SELECT 
    d.stringValue as item1,
    c.stringValue as item2,
    1 - VectorDistance('cosine', d.vectorValue, c.vectorValue) as cosineSimilarity
FROM c as d
JOIN c 
WHERE d.stringValue = 'Dog' AND c.stringValue = 'Cat'
"""

dog_cat_results = list(container.query_items(query=dog_cat_query, enable_cross_partition_query=True))
if dog_cat_results:
    result = dog_cat_results[0]
    print(f"Cosine similarity between {result['item1']} and {result['item2']}: {result['cosineSimilarity']:.6f}")

# Calculate similarity between Apple and Banana
print("\n=== Similarity: Apple vs Banana ===")
apple_banana_query = """
SELECT 
    d.stringValue as item1,
    c.stringValue as item2,
    1 - VectorDistance('cosine', d.vectorValue, c.vectorValue) as cosineSimilarity
FROM c as d
JOIN c 
WHERE d.stringValue = 'Apple' AND c.stringValue = 'Banana'
"""

apple_banana_results = list(container.query_items(query=apple_banana_query, enable_cross_partition_query=True))
if apple_banana_results:
    result = apple_banana_results[0]
    print(f"Cosine similarity between {result['item1']} and {result['item2']}: {result['cosineSimilarity']:.6f}")

# Calculate similarity between Dog and Banana
print("\n=== Similarity: Dog vs Banana ===")
dog_banana_query = """
SELECT 
    d.stringValue as item1,
    c.stringValue as item2,
    1 - VectorDistance('cosine', d.vectorValue, c.vectorValue) as cosineSimilarity
FROM c as d
JOIN c 
WHERE d.stringValue = 'Dog' AND c.stringValue = 'Banana'
"""

dog_banana_results = list(container.query_items(query=dog_banana_query, enable_cross_partition_query=True))
if dog_banana_results:
    result = dog_banana_results[0]
    print(f"Cosine similarity between {result['item1']} and {result['item2']}: {result['cosineSimilarity']:.6f}")

# Calculate similarity between Cilantro and Coriander (should be identical)
print("\n=== Similarity: Cilantro vs Coriander ===")
cilantro_coriander_query = """
SELECT 
    d.stringValue as item1,
    c.stringValue as item2,
    1 - VectorDistance('cosine', d.vectorValue, c.vectorValue) as cosineSimilarity
FROM c as d
JOIN c 
WHERE d.stringValue = 'Cilantro' AND c.stringValue = 'Coriander'
"""

cilantro_coriander_results = list(container.query_items(query=cilantro_coriander_query, enable_cross_partition_query=True))
if cilantro_coriander_results:
    result = cilantro_coriander_results[0]
    print(f"Cosine similarity between {result['item1']} and {result['item2']}: {result['cosineSimilarity']:.6f}")

# Find items most similar to Coriander
print("\n=== Items Most Similar to Coriander ===")

# First, get Coriander's vector
coriander_vector_query = "SELECT c.vectorValue FROM c WHERE c.stringValue = 'Coriander'"
coriander_result = list(container.query_items(query=coriander_vector_query, enable_cross_partition_query=True))

if coriander_result:
    coriander_vector = coriander_result[0]['vectorValue']
    
    # Find most similar items
    similarity_query = """
    SELECT 
        c.stringValue,
        1 - VectorDistance('cosine', c.vectorValue, @corianderVector) as cosineSimilarity 
    FROM c
    ORDER BY VectorDistance('cosine', c.vectorValue, @corianderVector)
    """
    
    similarity_results = list(container.query_items(
        query=similarity_query,
        parameters=[
            {"name": "@corianderVector", "value": coriander_vector}
        ],
        enable_cross_partition_query=True
    ))
    
    print("Items ranked by similarity to Coriander:")
    for i, item in enumerate(similarity_results, 1):
        print(f"{i}. {item['stringValue']}: {item['cosineSimilarity']:.6f}")

# Cleanup (uncomment to delete test data)
print("\n=== Cleanup ===")
print("To clean up test data, uncomment the cleanup section below")

# # Delete all test items
# print("Deleting test data...")
# for item in test_items:
#     try:
#         container.delete_item(item=item['id'], partition_key=item['stringValue'])
#         print(f"Deleted: {item['stringValue']}")
#     except Exception as e:
#         print(f"Error deleting {item['stringValue']}: {e}")

# # Delete container
# database.delete_container(container)
# print(f"Container '{CONTAINER_NAME}' deleted")

# # Delete database  
# client.delete_database(database)
# print(f"Database '{DATABASE_NAME}' deleted")

print("\nDemo completed!")

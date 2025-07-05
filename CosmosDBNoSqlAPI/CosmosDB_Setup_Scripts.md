# ========================================
# Azure Cosmos DB NoSQL API Setup Scripts
# Movie Dataset Vector Database Demo
# ========================================

# This file contains the configuration and setup scripts for creating
# a Cosmos DB vector database equivalent to the Azure SQL Database setup

## Database and Container Configuration

### 1. Database Creation
```python
from azure.cosmos import CosmosClient, PartitionKey

# Initialize Cosmos DB client
cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)

# Create database
database = cosmos_client.create_database_if_not_exists(id="MovieVectorDB")
```

### 2. Container Creation with Vector Capabilities
```python
# Define vector embedding policy for 1536-dimensional vectors
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
    "includedPaths": [
        {
            "path": "/*"
        }
    ],
    "excludedPaths": [
        {
            "path": "/embedding/*"
        }
    ],
    "vectorIndexes": [
        {
            "path": "/embedding",
            "type": "quantizedFlat"
        }
    ]
}

# Create container with vector search capabilities
container = database.create_container_if_not_exists(
    id="movies",
    partition_key=PartitionKey(path="/movie_id"),
    indexing_policy=indexing_policy,
    vector_embedding_policy=vector_embedding_policy,
    offer_throughput=1000  # Set appropriate RU/s
)
```

## Document Schema

### Movie Document Structure
```json
{
    "id": "123",                    // Cosmos DB document id (string)
    "movie_id": 123,               // Partition key (number)
    "title": "Sample Movie",
    "overview": "Movie description...",
    "genres": "[{\"id\": 28, \"name\": \"Action\"}]",
    "release_date": "2020-01-01",
    "budget": 100000000,
    "revenue": 250000000,
    "runtime": 120.0,
    "vote_average": 7.5,
    "vote_count": 1500,
    "popularity": 85.5,
    "original_language": "en",
    "combined_text": "Sample Movie Movie description... Action",
    "embedding": [0.001, 0.002, ...], // 1536-dimensional vector
    "embedding_model": "text-embedding-ada-002",
    "created_at": "2024-01-01T00:00:00Z",
    "document_type": "movie"
}
```

## Sample Data Insertion

### Insert Sample Movies with Vectors
```python
# Sample movie documents
sample_movies = [
    {
        "id": "1",
        "movie_id": 1,
        "title": "Test Action Movie",
        "overview": "A thrilling action adventure movie",
        "genres": "[{\"id\": 28, \"name\": \"Action\"}, {\"id\": 12, \"name\": \"Adventure\"}]",
        "release_date": "2020-01-01",
        "budget": 100000000,
        "revenue": 250000000,
        "runtime": 120.0,
        "vote_average": 7.5,
        "vote_count": 1500,
        "popularity": 85.5,
        "original_language": "en",
        "combined_text": "Test Action Movie A thrilling action adventure movie Action Adventure",
        "embedding": [0.001] * 1536,  # Sample 1536-dimensional vector
        "embedding_model": "text-embedding-ada-002",
        "created_at": "2024-01-01T00:00:00Z",
        "document_type": "movie"
    },
    {
        "id": "2",
        "movie_id": 2,
        "title": "Test Romance Movie",
        "overview": "A romantic comedy for the whole family",
        "genres": "[{\"id\": 35, \"name\": \"Comedy\"}, {\"id\": 10749, \"name\": \"Romance\"}]",
        "release_date": "2019-06-15",
        "budget": 50000000,
        "revenue": 150000000,
        "runtime": 95.0,
        "vote_average": 6.8,
        "vote_count": 800,
        "popularity": 42.3,
        "original_language": "en",
        "combined_text": "Test Romance Movie A romantic comedy for the whole family Comedy Romance",
        "embedding": [0.002] * 1536,  # Sample 1536-dimensional vector
        "embedding_model": "text-embedding-ada-002",
        "created_at": "2024-01-01T00:00:00Z",
        "document_type": "movie"
    },
    {
        "id": "3",
        "movie_id": 3,
        "title": "Test Family Movie",
        "overview": "An animated family film with adventure",
        "genres": "[{\"id\": 16, \"name\": \"Animation\"}, {\"id\": 10751, \"name\": \"Family\"}]",
        "release_date": "2021-03-20",
        "budget": 75000000,
        "revenue": 200000000,
        "runtime": 105.0,
        "vote_average": 8.2,
        "vote_count": 2200,
        "popularity": 91.7,
        "original_language": "en",
        "combined_text": "Test Family Movie An animated family film with adventure Animation Family",
        "embedding": [0.003] * 1536,  # Sample 1536-dimensional vector
        "embedding_model": "text-embedding-ada-002",
        "created_at": "2024-01-01T00:00:00Z",
        "document_type": "movie"
    }
]

# Insert sample data
for movie in sample_movies:
    container.create_item(movie)
```

## Vector Search Queries

### 1. Basic Vector Similarity Search
```python
# Search for similar movies using VectorDistance
query = """
SELECT TOP 10 
    c.title, 
    c.overview, 
    c.vote_average,
    VectorDistance(c.embedding, @queryVector) AS similarity_score
FROM c 
WHERE c.document_type = 'movie' AND IS_DEFINED(c.embedding)
ORDER BY VectorDistance(c.embedding, @queryVector)
"""

results = container.query_items(
    query=query,
    parameters=[
        {"name": "@queryVector", "value": query_vector}
    ],
    enable_cross_partition_query=True
)
```

### 2. Filtered Vector Search
```python
# Search with additional filters
query = """
SELECT TOP 5 
    c.title, 
    c.overview, 
    c.vote_average,
    c.popularity,
    VectorDistance(c.embedding, @queryVector) AS similarity_score
FROM c 
WHERE c.document_type = 'movie' 
    AND IS_DEFINED(c.embedding)
    AND c.vote_average >= @minRating
    AND c.popularity >= @minPopularity
ORDER BY VectorDistance(c.embedding, @queryVector)
"""

results = container.query_items(
    query=query,
    parameters=[
        {"name": "@queryVector", "value": query_vector},
        {"name": "@minRating", "value": 7.0},
        {"name": "@minPopularity", "value": 50.0}
    ],
    enable_cross_partition_query=True
)
```

### 3. Hybrid Search (Vector + Text)
```python
# Combine vector similarity with text search
query = """
SELECT TOP 10 
    c.title, 
    c.overview, 
    c.vote_average,
    VectorDistance(c.embedding, @queryVector) AS similarity_score
FROM c 
WHERE c.document_type = 'movie' 
    AND IS_DEFINED(c.embedding)
    AND (
        CONTAINS(UPPER(c.title), UPPER(@searchText))
        OR CONTAINS(UPPER(c.overview), UPPER(@searchText))
        OR CONTAINS(UPPER(c.genres), UPPER(@searchText))
    )
ORDER BY VectorDistance(c.embedding, @queryVector)
"""

results = container.query_items(
    query=query,
    parameters=[
        {"name": "@queryVector", "value": query_vector},
        {"name": "@searchText", "value": "action"}
    ],
    enable_cross_partition_query=True
)
```

## Indexing and Performance

### Vector Index Configuration
```python
# Vector index types available:
# - "quantizedFlat": Good balance of performance and accuracy
# - "diskANN": Better for larger datasets (when available)

vector_indexes = [
    {
        "path": "/embedding",
        "type": "quantizedFlat"
    }
]
```

### Performance Optimization
```python
# 1. Partition key strategy
# - Use movie_id as partition key for even distribution
# - Consider genre or year for specific query patterns

# 2. Query optimization
# - Always use TOP N to limit results
# - Enable cross-partition queries when needed
# - Use point reads when possible (by id and partition key)

# 3. Throughput settings
# - Use serverless for unpredictable workloads
# - Use provisioned for predictable workloads
# - Consider autoscale for variable workloads
```

## Monitoring and Metrics

### RU Consumption Monitoring
```python
# Monitor RU consumption for vector queries
for item in container.query_items(
    query=vector_search_query,
    parameters=parameters,
    enable_cross_partition_query=True
):
    # Access RU charge from response headers
    pass
```

### Container Metrics
```python
# Get container properties
container_properties = container.read()

# Check vector embedding policy
vector_policy = container_properties.get('vectorEmbeddingPolicy', {})
print(f"Vector embeddings configured: {len(vector_policy.get('vectorEmbeddings', []))}")

# Check indexing policy
indexing_policy = container_properties.get('indexingPolicy', {})
print(f"Vector indexes configured: {len(indexing_policy.get('vectorIndexes', []))}")
```

## Cleanup Scripts

### Delete All Documents
```python
# Delete all movie documents
all_docs = container.query_items(
    query="SELECT c.id, c.movie_id FROM c WHERE c.document_type = 'movie'",
    enable_cross_partition_query=True
)

for doc in all_docs:
    container.delete_item(item=doc['id'], partition_key=doc['movie_id'])
```

### Delete Container
```python
# Delete the entire container
database.delete_container(container)
```

### Delete Database
```python
# Delete the entire database
cosmos_client.delete_database(database)
```

## Environment Variables Required

```bash
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_API_VERSION=2024-06-01
EMBEDDING_MODEL=text-embedding-ada-002
GENERATION_MODEL=gpt-4o

# Azure Cosmos DB Configuration
COSMOS_ENDPOINT=your_cosmos_endpoint
COSMOS_KEY=your_cosmos_key
COSMOS_DATABASE_NAME=MovieVectorDB
COSMOS_CONTAINER_NAME=movies
```

## Key Differences from SQL Database

1. **Document-based**: Uses JSON documents instead of relational tables
2. **Partition Key**: Required for horizontal scaling
3. **Vector Index Types**: Different index types (quantizedFlat vs SQL vector indexes)
4. **Query Language**: SQL-like but with document-specific functions
5. **Consistency Models**: Multiple consistency levels available
6. **Global Distribution**: Built-in multi-region capabilities
7. **Serverless Option**: Pay-per-request pricing model available
8. **Change Feed**: Real-time change tracking capabilities

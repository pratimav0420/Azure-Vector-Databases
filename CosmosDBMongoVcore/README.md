# Azure Cosmos DB for MongoDB vCore - Vector Database Demo

This folder contains comprehensive examples and demonstrations of vector database operations using **Azure Cosmos DB for MongoDB vCore** with MongoDB's native vector search capabilities.

## 📋 Overview

Azure Cosmos DB for MongoDB vCore provides native vector search capabilities through MongoDB's `$vectorSearch` aggregation stage, allowing you to perform semantic similarity searches on high-dimensional embeddings alongside traditional MongoDB operations.

### Key Features Demonstrated

- **Native Vector Search**: Using MongoDB's `$vectorSearch` aggregation pipeline
- **Vector Indexing**: SearchIndexModel for optimized vector operations
- **Filtered Vector Search**: Combining vector similarity with traditional filters
- **Azure OpenAI Integration**: Text embeddings and chat completions
- **Advanced Vector Analytics**: Clustering, similarity analysis, and performance benchmarking
- **Hybrid Search**: Combining vector and traditional text search
- **RAG Implementation**: Complete Retrieval-Augmented Generation pipeline

## 📁 Files Structure

```
CosmosDBMongoVcore/
├── README.md                                    # This file
├── MovieDataset_MongoVcore_Demo.ipynb         # Main comprehensive notebook
├── MongoDB Demo 1.1 - Simple Vector Query.py  # Basic vector operations
├── MongoDB Demo 1.2 - Simple Embedding.py     # Embedding generation and search
├── MongoDB Demo 2.1 - Simple Completion.py    # Chat completion with history
├── MongoDB Demo 2.2 - End to End RAG.py      # Complete RAG implementation
└── Advanced_Vector_Operations.py              # Advanced vector analytics
```

## 🚀 Quick Start

### Prerequisites

1. **Azure Cosmos DB for MongoDB vCore** account
2. **Azure OpenAI** service with deployed models
3. **Python 3.8+** with required packages

### Environment Setup

1. **Install Dependencies**:
   ```bash
   pip install pymongo motor openai python-dotenv numpy pandas scikit-learn matplotlib seaborn
   ```

2. **Configure Environment Variables**:
   Create a `.env` file in the project root:
   ```env
   # MongoDB vCore Connection
   MONGODB_CONNECTION_STRING=mongodb+srv://<username>:<password>@<cluster>.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000

   # Azure OpenAI Configuration
   AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
   AZURE_OPENAI_API_KEY=<your-api-key>
   AZURE_OPENAI_API_VERSION=2024-06-01
   EMBEDDING_MODEL=text-embedding-ada-002
   COMPLETION_MODEL=gpt-4o
   ```

3. **Create MongoDB vCore Resource**:
   - Create Azure Cosmos DB for MongoDB vCore account
   - Note the connection string format specific to vCore
   - Ensure vector search capabilities are enabled

## 📚 Demo Scripts Overview

### 1. Simple Vector Query (`MongoDB Demo 1.1`)
**Purpose**: Basic vector operations with 2D vectors for learning
- Manual vector storage and retrieval
- Cosine similarity calculations
- Fallback search without vector indexes
- Simple vector mathematics demonstrations

**Key Operations**:
```python
# Store vectors manually
collection.insert_one({
    "title": "Action Movie",
    "vector": [0.8, 0.6],  # 2D vector for simplicity
    "category": "movie"
})

# Simple cosine similarity search
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
```

### 2. Simple Embedding (`MongoDB Demo 1.2`)
**Purpose**: Real embeddings from Azure OpenAI with vector indexing
- Azure OpenAI embedding generation (1536 dimensions)
- Vector index creation with SearchIndexModel
- Semantic similarity search using `$vectorSearch`
- Embedding model integration and optimization

**Key Operations**:
```python
# Create vector search index
vector_index = SearchIndexModel(
    definition={
        "fields": [{
            "type": "vector",
            "path": "embedding",
            "numDimensions": 1536,
            "similarity": "cosine"
        }]
    },
    name="vector_index"
)

# Vector search aggregation
pipeline = [{
    "$vectorSearch": {
        "index": "vector_index",
        "path": "embedding", 
        "queryVector": query_embedding,
        "numCandidates": 100,
        "limit": 10
    }
}]
```

### 3. Simple Completion (`MongoDB Demo 2.1`)
**Purpose**: Chat completions with conversation history
- GPT-4o integration for chat completions
- Conversation history storage in MongoDB
- Multi-turn conversation management
- Context building and conversation analytics

**Key Operations**:
```python
# Store conversation history
conversation_doc = {
    "conversation_id": str(uuid.uuid4()),
    "messages": [],
    "created_at": datetime.utcnow()
}

# Generate completion with history
completion = openai_client.chat.completions.create(
    model=COMPLETION_MODEL,
    messages=conversation_history
)
```

### 4. End-to-End RAG (`MongoDB Demo 2.2`)
**Purpose**: Complete Retrieval-Augmented Generation system
- Knowledge base creation and management
- Semantic document retrieval
- Context-aware answer generation
- RAG performance analytics and optimization
- Knowledge base updates and maintenance

**Key Operations**:
```python
# RAG pipeline implementation
def rag_query(question):
    # 1. Generate question embedding
    question_embedding = get_embedding(question)
    
    # 2. Retrieve relevant documents
    relevant_docs = vector_search(question_embedding)
    
    # 3. Generate answer with context
    context = "\n".join([doc['content'] for doc in relevant_docs])
    answer = generate_completion(question, context)
    
    return answer, relevant_docs
```

### 5. Advanced Vector Operations (`Advanced_Vector_Operations.py`)
**Purpose**: Production-ready advanced vector analytics
- Multi-dimensional vector clustering (K-means)
- Similarity matrix analysis across categories
- Temporal vector pattern analysis
- Hybrid search (vector + traditional)
- Performance benchmarking
- Collection statistics and optimization

**Advanced Features**:
```python
# Advanced filtered vector search
def filtered_vector_search(query, filters):
    pipeline = [{
        "$vectorSearch": {
            "index": "advanced_vector_index",
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": 100,
            "limit": top_k,
            "filter": {
                "category": {"$eq": category},
                "year": {"$gte": start_year, "$lte": end_year},
                "rating": {"$gte": min_rating}
            }
        }
    }]
```

## 🎯 MongoDB vCore Vector Search Features

### Vector Index Types
- **IVF (Inverted File)**: For large-scale approximate search
- **HNSW (Hierarchical Navigable Small World)**: For high-accuracy search
- **Cosine Similarity**: Default similarity metric for normalized vectors

### Aggregation Pipeline Integration
```python
# Complex aggregation with vector search
pipeline = [
    {"$vectorSearch": {
        "index": "vector_index",
        "path": "embedding",
        "queryVector": query_vector,
        "numCandidates": 100,
        "limit": 50
    }},
    {"$match": {"category": "movie"}},
    {"$group": {
        "_id": "$genre",
        "avg_score": {"$avg": {"$meta": "vectorSearchScore"}},
        "count": {"$sum": 1}
    }},
    {"$sort": {"avg_score": -1}}
]
```

### Filter Integration
```python
# Vector search with complex filters
filter_conditions = {
    "$and": [
        {"year": {"$gte": 2000}},
        {"rating": {"$gte": 8.0}},
        {"$or": [
            {"genre": "sci-fi"},
            {"genre": "action"}
        ]}
    ]
}
```

## 📈 Performance Optimization

### Index Optimization
- Use appropriate `numCandidates` values (typically 10x your limit)
- Optimize vector dimensions for your use case
- Consider index build time vs search performance trade-offs

### Query Optimization
```python
# Optimized vector search
{
    "$vectorSearch": {
        "index": "optimized_vector_index",
        "path": "embedding",
        "queryVector": query_vector,
        "numCandidates": 200,  # 10x limit for better recall
        "limit": 20,
        "filter": {"category": {"$eq": "movie"}}  # Pre-filter for efficiency
    }
}
```

### Best Practices
1. **Batch Operations**: Use `insert_many()` for bulk vector inserts
2. **Connection Pooling**: Reuse MongoDB connections
3. **Index Monitoring**: Monitor index usage and performance
4. **Embedding Caching**: Cache embeddings for repeated queries
5. **Async Operations**: Use `motor` for async MongoDB operations

## 🔧 Troubleshooting

### Common Issues

1. **Connection String Format**:
   ```
   # Correct MongoDB vCore format
   mongodb+srv://username:password@cluster.mongocluster.cosmos.azure.com/
   ```

2. **Vector Index Creation**:
   ```python
   # Wait for index to be ready
   time.sleep(30)  # Index creation can take time
   ```

3. **Embedding Dimensions**:
   ```python
   # Ensure consistent dimensions
   "numDimensions": 1536  # Must match your embedding model
   ```

### Performance Tips

1. **Monitor Vector Search Performance**:
   ```python
   # Use explain to analyze query performance
   explain_result = collection.aggregate(pipeline, explain=True)
   ```

2. **Optimize numCandidates**:
   ```python
   # Balance between accuracy and performance
   "numCandidates": min(100, num_documents)  # Don't exceed document count
   ```

## 🎮 Interactive Demo

Run the main notebook for an interactive experience:
```bash
jupyter notebook MovieDataset_MongoVcore_Demo.ipynb
```

The notebook provides step-by-step guidance through:
1. Environment setup and configuration
2. Connection establishment
3. Vector index creation and management
4. Sample data ingestion with embeddings
5. Various vector search scenarios
6. Performance analysis and optimization
7. Clean-up procedures

## 🚀 Production Deployment

### Scalability Considerations
- **Sharding Strategy**: Design for horizontal scaling
- **Index Distribution**: Distribute vector indexes across shards
- **Connection Management**: Use connection pooling
- **Monitoring**: Implement comprehensive monitoring

### Security Best Practices
- Use Azure managed identities when possible
- Implement network security rules
- Enable encryption in transit and at rest
- Regular security audits and updates

### Cost Optimization
- Monitor RU consumption for vector operations
- Optimize embedding dimensions
- Use appropriate consistency levels
- Implement data lifecycle management

## 📖 Additional Resources

- [Azure Cosmos DB for MongoDB vCore Documentation](https://docs.microsoft.com/azure/cosmos-db/mongodb/)
- [MongoDB Vector Search Documentation](https://docs.mongodb.com/atlas/atlas-vector-search/)
- [Azure OpenAI Service Documentation](https://docs.microsoft.com/azure/cognitive-services/openai/)
- [Vector Database Best Practices](https://docs.microsoft.com/azure/architecture/guide/technology-choices/vector-search)

## 🤝 Contributing

Feel free to contribute improvements, additional examples, or optimizations to these demos. Areas of interest:
- Performance benchmarking scripts
- Additional embedding models integration
- Advanced RAG techniques
- Production monitoring examples
- Cost optimization strategies

---

*This demo showcases Azure Cosmos DB for MongoDB vCore's vector search capabilities. For production use, consider additional security, monitoring, and optimization measures based on your specific requirements.*

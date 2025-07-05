# Azure Cosmos DB NoSQL API Vector Database Demo

This folder contains scripts and notebooks demonstrating how to use Azure Cosmos DB NoSQL API as a vector database for movie recommendations and semantic search.

## 📁 Contents

### Core Notebook
- **`MovieDataset_CosmosDBVectors_Demo.ipynb`** - Complete demo notebook showing vector database operations with Cosmos DB

### Demo Scripts
- **`CosmosDB Demo 1.1 - Simple Vector Query.py`** - Basic vector operations and similarity calculations
- **`CosmosDB Demo 1.2 - Simple Embedding.py`** - Embedding generation and semantic search examples
- **`CosmosDB Demo 2.1 - Simple Completion.py`** - Azure OpenAI chat completion with conversation history
- **`CosmosDB Demo 2.2 - End to End RAG.py`** - Complete RAG implementation using Cosmos DB as vector store

### Advanced Operations
- **`Advanced_Vector_Operations.py`** - Advanced vector operations including multi-vector search, clustering analysis, and performance benchmarks

### Setup Documentation
- **`CosmosDB_Setup_Scripts.md`** - Detailed setup instructions and configuration examples

## 🚀 Quick Start

### Prerequisites

1. **Azure Cosmos DB Account** with NoSQL API and vector search enabled
2. **Azure OpenAI Service** with text-embedding-ada-002 and gpt-4o models deployed
3. **Python 3.8+** with required packages

### Environment Setup

1. **Copy environment template:**
   ```bash
   cp ../.env.example ../.env
   ```

2. **Configure your credentials in `.env`:**
   ```bash
   # Azure OpenAI Configuration
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_API_KEY=your_openai_api_key
   AZURE_OPENAI_API_VERSION=2024-06-01
   EMBEDDING_MODEL=text-embedding-ada-002
   GENERATION_MODEL=gpt-4o

   # Azure Cosmos DB Configuration
   COSMOS_ENDPOINT=https://your-cosmos-account.documents.azure.com:443/
   COSMOS_KEY=your_cosmos_primary_key
   COSMOS_DATABASE_NAME=MovieVectorDB
   COSMOS_CONTAINER_NAME=movies
   ```

3. **Install dependencies:**
   ```bash
   pip install -r ../requirements.txt
   ```

### Running the Demos

#### Option 1: Jupyter Notebook (Recommended)
```bash
jupyter notebook MovieDataset_CosmosDBVectors_Demo.ipynb
```

#### Option 2: Python Scripts
```bash
# Basic vector operations
python "CosmosDB Demo 1.1 - Simple Vector Query.py"

# Embedding and semantic search
python "CosmosDB Demo 1.2 - Simple Embedding.py"

# Chat completion with history
python "CosmosDB Demo 2.1 - Simple Completion.py"

# Complete RAG system
python "CosmosDB Demo 2.2 - End to End RAG.py"

# Advanced vector operations
python "Advanced_Vector_Operations.py"
```

## 🎯 What You'll Learn

### Vector Database Concepts
- **Document-based vector storage** in Cosmos DB JSON documents
- **Multiple vector embeddings** per document (content, title, genre)
- **Vector similarity search** using VectorDistance function
- **Hybrid search** combining vector and traditional queries

### Azure Cosmos DB Features
- **Container setup** with vector embedding policies
- **Vector indexing** with quantizedFlat index type
- **Cross-partition queries** for vector search
- **Global distribution** and scaling capabilities
- **RU consumption** monitoring for vector operations

### Practical Applications
- **Movie recommendation system** with semantic search
- **RAG (Retrieval Augmented Generation)** with conversation history
- **Multi-modal search** across different content types
- **Real-time vector updates** using change feed

## 📊 Key Differences from SQL Database Approach

| Aspect | Cosmos DB NoSQL API | Azure SQL Database |
|--------|-------------------|-------------------|
| **Data Model** | JSON documents | Relational tables |
| **Vector Storage** | Array property in document | VECTOR data type |
| **Partitioning** | Required partition key | Optional |
| **Scaling** | Automatic horizontal scaling | Manual scaling |
| **Consistency** | Multiple consistency levels | ACID transactions |
| **Pricing** | RU-based or serverless | DTU/vCore based |
| **Global Distribution** | Built-in multi-region | Manual setup |
| **Query Language** | SQL-like with JSON functions | Standard T-SQL |

## 🔧 Advanced Features Demonstrated

### Multi-Vector Search
```python
# Search across different embedding types
content_results = search_by_content("action movie")
title_results = search_by_title("matrix")
genre_results = search_by_genre("sci-fi")
```

### Hybrid Vector Scoring
```python
# Combine multiple vector similarities with weights
hybrid_search(query, weights={
    "content": 0.6,
    "title": 0.3,
    "genre": 0.1
})
```

### Vector Clustering Analysis
```python
# Analyze similarity patterns between movies
clustering_analysis()
# Output: Movie similarity matrix and genre clustering
```

### Performance Optimization
```python
# Benchmark vector operations
performance_test()
# Output: Query latency, RU consumption, throughput metrics
```

## 📈 Performance Characteristics

### Vector Query Performance
- **Simple similarity search**: ~10-50ms (depending on dataset size)
- **Complex filtered search**: ~20-100ms
- **Multi-vector comparison**: ~50-200ms
- **Hybrid search**: ~100-500ms (application-level aggregation)

### Scalability
- **Documents**: Virtually unlimited with auto-partitioning
- **Vector dimensions**: Up to 2000 per vector path
- **Concurrent queries**: Scales with provisioned RU/s
- **Global distribution**: <100ms cross-region replication

### Cost Optimization
- **Serverless**: Pay per operation (good for unpredictable workloads)
- **Provisioned**: Fixed RU/s (good for predictable workloads)
- **Autoscale**: Automatic scaling within bounds
- **Vector RU consumption**: ~5-20 RUs per similarity query

## 🛠️ Production Considerations

### Security
- Use **Azure Key Vault** for connection strings and keys
- Implement **Azure AD authentication** when possible
- Enable **network access controls** and private endpoints
- Regular **key rotation** for Cosmos DB and OpenAI keys

### Monitoring
- **Azure Monitor** for Cosmos DB metrics
- **Application Insights** for application telemetry
- **Custom logging** for vector search analytics
- **RU consumption** tracking and alerting

### High Availability
- **Multi-region writes** for global applications
- **Automatic failover** with single region writes
- **Change feed** for real-time data processing
- **Backup and restore** strategies

### Performance Tuning
- **Partition key strategy** for even distribution
- **Indexing policy** optimization for your queries
- **Connection pooling** for high-throughput scenarios
- **Batch operations** for bulk data loading

## 🔍 Troubleshooting

### Common Issues

1. **Vector search returns no results**
   - Check vector embedding policy configuration
   - Verify vector index is created
   - Ensure embeddings are properly stored

2. **High RU consumption**
   - Optimize partition key strategy
   - Use TOP N queries to limit results
   - Consider caching for repeated queries

3. **Cross-partition query limitations**
   - Enable cross-partition queries explicitly
   - Consider partition key design for your access patterns

4. **Embedding generation failures**
   - Check Azure OpenAI quota and rate limits
   - Implement retry logic with exponential backoff
   - Monitor API key validity

### Performance Issues
- **Slow vector queries**: Check indexing policy and partition distribution
- **High latency**: Consider global distribution and region selection
- **RU throttling**: Scale up provisioned throughput or use autoscale

## 📚 Additional Resources

- [Azure Cosmos DB Vector Search Documentation](https://docs.microsoft.com/azure/cosmos-db/vector-search)
- [Azure OpenAI Service Documentation](https://docs.microsoft.com/azure/cognitive-services/openai/)
- [Cosmos DB Best Practices](https://docs.microsoft.com/azure/cosmos-db/best-practice-guide)
- [Vector Database Design Patterns](https://docs.microsoft.com/azure/architecture/guide/vector-database-patterns)

## 🤝 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review Azure Cosmos DB documentation
3. Submit issues to the repository
4. Consult Azure support for service-specific issues

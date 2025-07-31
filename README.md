# Azure Vector Databases

## 🧭 Goal
This project provides hands-on experience with multiple **Vector Database** options available in **Azure**. It demonstrates how to use Azure SQL Database, Azure Cosmos DB NoSQL API, Azure Cosmos DB for MongoDB vCore, Azure Database for PostgreSQL, and Microsoft Fabric EventHouse as vector stores with Azure OpenAI embeddings for building modern AI applications including semantic search and RAG (Retrieval Augmented Generation) systems.


## 🎯 Who This Is For

This repository is useful for:
- **Developers** building AI-powered applications with vector search
- **Data engineers** implementing semantic search pipelines
- **AI/ML practitioners** integrating Azure OpenAI with vector databases
- **Solution architects** designing RAG systems on Azure
- **Students** learning vector databases and modern AI applications

---

## 🚀 Getting Started

> 🏃‍♂️ **Want to get started quickly?** Check out our [**Quick Start Guide**](QUICKSTART.md) for a 5-minute setup!

### Prerequisites
- **Azure Subscription** with access to:
  - Azure SQL Database (with vector support enabled)
  - Azure Cosmos DB (NoSQL API and MongoDB vCore)
  - Azure Database for PostgreSQL (Flexible Server with pgvector)
  - Microsoft Fabric workspace (with EventHouse)
  - Azure OpenAI Service (with text-embedding-3-small and gpt-4o-mini deployments)
- **Python 3.10+** installed
- **Git** for cloning the repository

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/azure-vector-databases.git
cd azure-vector-databases
```

### 2. Install Python Dependencies
```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 3. Configure Environment Variables
```bash
# Copy the environment template
cp .env.example .env

# Edit .env with your actual credentials
# Use your preferred text editor to fill in:
# - Azure OpenAI endpoint and API key
# - Azure SQL Database connection details
```

**🔐 Security Note**: The `.env` file contains sensitive credentials and should never be committed to version control.

### 4. Set Up Azure Resources

#### 3.1 Azure SQL Database
1. **Create an Azure SQL Database** with vector support enabled
2. **Note your connection details**:
   - Server name (e.g., `your-server.database.windows.net`)
   - Database name
   - Username and password

#### 3.2 Azure OpenAI Service
1. **Create an Azure OpenAI resource** in Azure Portal
2. **Deploy the required models**:
   - `text-embedding-3-small` (for embeddings)
   - `gpt-4o-mini` (for text generation)
3. **Note your configuration**:
   - Endpoint URL
   - API key
   - Deployment names

### 4. Configure the Application
```bash
# Copy the configuration template
cp config_template.py config.py

# Edit config.py with your actual Azure credentials
# IMPORTANT: Never commit config.py to version control
```

Update `config.py` with your Azure credentials:
```python
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"
AZURE_OPENAI_API_KEY = "your-api-key"

# Azure SQL Database Configuration
SERVER = "your-server.database.windows.net"
DATABASE = "your-database-name"
USERNAME = "your-username"
PASSWORD = "your-password"
```

### 5. Set Up the Database Schema
```bash
# Connect to your Azure SQL Database and run:
# AzureSqlDB/MovieDB_DDL_Scripts.sql

# This will create:
# - movies table (for movie metadata)
# - movie_vectors table (for embeddings with VECTOR(1536) column)
# - Appropriate indexes for performance
```

### 6. Download the Dataset
The movies dataset should be placed in `data/moviesdataset/` directory. You can download it from:
- [The Movies Dataset on Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

Required files:
- `movies_metadata.csv`
- `ratings_small.csv`

### 6. Validate Your Setup
```bash
# Run the environment validation script
python setup.py

# This will check:
# ✅ Python version and dependencies
# ✅ Azure OpenAI connectivity
# ✅ Azure SQL Database connectivity
# ✅ Dataset availability
# ✅ Configuration completeness
```

If all checks pass, you're ready to start the demo! 🎉



## 📓 Usage Guide

### Option 1: Jupyter Notebook (Recommended)
```bash
# Start Jupyter Notebook
jupyter notebook

# Open MovieDataset_VectorDB_Demo.ipynb
# Follow the step-by-step guide in the notebook
```

The notebook provides a complete walkthrough:
1. **Data Loading** - Load and preprocess the movies dataset
2. **Database Setup** - Create tables with vector columns
3. **Embedding Generation** - Use Azure OpenAI to create embeddings
4. **Vector Search** - Implement semantic search functionality
5. **RAG Implementation** - Build a complete recommendation system
6. **Performance Analysis** - Optimize for production use

### Option 2: SQL Scripts
```bash
# Run the DDL scripts first
# Execute: AzureSqlDB/MovieDB_DDL_Scripts.sql

# Then explore advanced vector operations
# Execute: AzureSqlDB/Advanced_Vector_Operations.sql
```

### Option 3: Step-by-Step Manual Setup
1. **Create database tables** using `MovieDB_DDL_Scripts.sql`
2. **Load movie data** into the movies table
3. **Generate embeddings** using Azure OpenAI API
4. **Insert vectors** into the movie_vectors table
5. **Run similarity queries** to test the setup

### Option 4: Azure Cosmos DB NoSQL API
```bash
# Navigate to Cosmos DB demo folder
cd CosmosDBNoSqlAPI/

# Start with the main notebook
jupyter notebook MovieDataset_CosmosDBVectors_Demo.ipynb

# Or run individual demo scripts
python "CosmosDB Demo 1.1 - Simple Vector Query.py"
python "CosmosDB Demo 1.2 - Simple Embedding.py"
python "CosmosDB Demo 2.1 - Simple Completion.py"
python "CosmosDB Demo 2.2 - End to End RAG.py"
```

### Option 5: Azure Cosmos DB for MongoDB vCore
```bash
# Navigate to MongoDB vCore demo folder
cd CosmosDBMongoVcore/

# Start with the main notebook
jupyter notebook MovieDataset_MongoVcore_Demo.ipynb

# Or run individual demo scripts
python "MongoDB Demo 1.1 - Simple Vector Query.py"
python "MongoDB Demo 1.2 - Simple Embedding.py"
python "MongoDB Demo 2.1 - Simple Completion.py"
python "MongoDB Demo 2.2 - End to End RAG.py"
python "Advanced_Vector_Operations.py"


## 🏢 Vector Database Comparison: Complete Azure Vector Solutions

| Feature | Azure SQL Database | Cosmos DB NoSQL API | Cosmos DB MongoDB vCore | Azure PostgreSQL | Fabric EventHouse |
|---------|------------------|--------------------------|------------------------------|------------------|-------------------|
| **Vector Storage** | Native VECTOR(n) data type | JSON arrays in documents | BSON arrays with vector indexes | pgvector extension types | KQL vector functions |
| **Similarity Functions** | VECTOR_DISTANCE() | VectorDistance() | $vectorSearch aggregation | <-> operator, cosine() | series_cosine_similarity() |
| **Indexing** | Built-in vector indexes | Vector embedding policies | SearchIndexModel (HNSW/IVF) | IVF-Flat, HNSW indexes | Columnar storage optimization |
| **Query Language** | T-SQL with vector functions | SQL API with vector functions | MongoDB aggregation pipeline | SQL with vector operators | KQL (Kusto Query Language) |
| **Scaling** | Vertical (scale up/down) | Horizontal (automatic) | Horizontal (sharding) | Vertical with read replicas | Horizontal (cluster scaling) |
| **Global Distribution** | Read replicas | Multi-region writes | Multi-region with sharding | Cross-region replication | Global Fabric deployment |
| **Consistency** | ACID transactions | Tunable consistency levels | MongoDB consistency model | ACID transactions | Eventual consistency |
| **Schema** | Fixed schema (tables) | Flexible schema (JSON) | Flexible schema (BSON) | Fixed schema (tables) | Schema-on-read |
| **Ecosystem** | SQL Server ecosystem | Azure Cosmos DB tools | MongoDB ecosystem | PostgreSQL ecosystem | Microsoft Fabric suite |
| **Real-time Analytics** | Limited | Moderate | Moderate | Limited | Excellent |
| **Best For** | Structured data with vectors | Document-based vector storage | MongoDB-native vector apps | PostgreSQL + vector hybrid | Real-time vector analytics |

### When to Choose Azure SQL Database:
- ✅ **Relational data model** fits your needs
- ✅ **ACID transactions** are required
- ✅ **Strong consistency** is critical
- ✅ **Existing SQL expertise** in your team
- ✅ **Complex joins** between vector and relational data

### When to Choose Cosmos DB NoSQL API:
- ✅ **Global distribution** across multiple regions
- ✅ **Automatic scaling** with unpredictable workloads
- ✅ **Flexible schema** for evolving data models
- ✅ **Multi-model applications** (documents + vectors)
- ✅ **Low latency** requirements worldwide

### When to Choose Cosmos DB for MongoDB vCore:
- ✅ **Existing MongoDB expertise** and applications
- ✅ **Rich aggregation pipelines** with vector search
- ✅ **Advanced vector indexing** requirements (HNSW/IVF)
- ✅ **MongoDB ecosystem** compatibility needed
- ✅ **Complex vector operations** with document queries
- ✅ **Familiar MongoDB drivers** and tooling

### When to Choose Azure Database for PostgreSQL:
- ✅ **PostgreSQL expertise** and existing applications
- ✅ **Open-source preference** with enterprise hosting
- ✅ **Advanced SQL capabilities** with vector operations
- ✅ **Cost-effective solution** for moderate scale
- ✅ **Hybrid relational-vector** workloads
- ✅ **Strong ACID compliance** requirements

### When to Choose Microsoft Fabric EventHouse:
- ✅ **Real-time vector analytics** and streaming data
- ✅ **Large-scale data processing** requirements
- ✅ **Microsoft ecosystem integration** (Power BI, Azure ML)
- ✅ **Time-series + vector analysis** combined workloads
- ✅ **Enterprise data governance** and compliance
- ✅ **Advanced analytics** with KQL capabilities

---

## �🎬 Demo Examples

### Semantic Movie Search
```python
# Search for movies using natural language
results = semantic_movie_search_azure_openai("action movies with explosions and car chases")

# Results will include movies like:
# - Fast & Furious series
# - Mission Impossible movies
# - Marvel action films
```

### RAG Movie Recommendations
```python
# Get personalized recommendations with explanations
recommendation = rag_movie_recommendation("I want a heartwarming family movie for the weekend")

# GPT-4o Mini will analyze retrieved movies and provide:
# - Personalized recommendations
# - Detailed explanations of why each movie fits
# - Information about plot, ratings, and genres
```

### Vector Similarity in SQL
```sql
-- Find movies similar to a specific movie
DECLARE @target_vector VECTOR(1536)
SELECT @target_vector = embedding FROM movie_vectors WHERE title = 'Toy Story'

SELECT TOP 5 
    title,
    1 - VECTOR_DISTANCE('cosine', embedding, @target_vector) AS similarity_score
FROM movie_vectors
ORDER BY similarity_score DESC
```

### Vector Similarity in Cosmos DB
```python
# Find movies similar to a specific movie using Cosmos DB
query = """
SELECT TOP 5 
    c.title,
    VectorDistance(c.embedding, @queryVector) AS distance
FROM c
WHERE c.embedding != null
ORDER BY VectorDistance(c.embedding, @queryVector)
"""

results = container.query_items(
    query=query,
    parameters=[{"name": "@queryVector", "value": target_embedding}],
    enable_cross_partition_query=True
)
```


## 🔧 Configuration Options



### Getting Help
- 🚀 [**Quick Start Guide**](QUICKSTART.md) - 5-minute setup
- 🛠️ [**Troubleshooting Guide**](TROUBLESHOOTING.md) - Common issues and solutions
- 📝 Check the [Issues](../../issues) section for common problems
- 📖 Review Azure SQL Database vector documentation
- 🤖 Consult Azure OpenAI service limits and quotas



## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and test thoroughly
4. **Commit your changes**: `git commit -m 'Add amazing feature'`
5. **Push to the branch**: `git push origin feature/amazing-feature`
6. **Open a Pull Request**

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Format code
black *.py

# Type checking
mypy *.py
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---


**Happy Vector Searching! 🚀**

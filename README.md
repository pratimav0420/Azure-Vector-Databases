# Azure Vector Databases

## 🧭 Goal
This project provides hands-on experience with multiple **Vector Database** options available in **Azure**. It demonstrates how to use Azure SQL Database as a vector store with Azure OpenAI embeddings for building modern AI applications including semantic search and RAG (Retrieval Augmented Generation) systems.

---

## 🗺️ High-Level Flow

- **Azure SQL Database** with native VECTOR data type support
- **Azure OpenAI** integration for embeddings (text-embedding-3-small) and generation (GPT-4o Mini)
- **Movie Dataset** as sample data for demonstrations
- **Complete RAG pipeline** implementation
- **Vector similarity search** with cosine and euclidean distance
- **Performance optimization** and cost management strategies

### Key Features Demonstrated:
- ✅ **1536-dimensional vectors** using Azure OpenAI text-embedding-3-small
- ✅ **Native SQL vector operations** (VECTOR_DISTANCE, similarity search)
- ✅ **Semantic movie search** with natural language queries
- ✅ **RAG implementation** with GPT-4o Mini for personalized recommendations
- ✅ **Production-ready patterns** with error handling and optimization

---

## 📁 Project Structure

```
Azure-Vector-Databases/
├── data/
│   └── moviesdataset/          # Movies dataset CSV files
├── AzureSqlDB/
│   ├── MovieDB_DDL_Scripts.sql        # Database schema creation
│   ├── Advanced_Vector_Operations.sql  # Advanced SQL vector demos
│   └── SQL Demo 1.1-2.2*.sql          # Basic vector examples
├── MovieDataset_VectorDB_Demo.ipynb   # Main Python notebook
├── config_template.py                 # Configuration template
├── .env.example                       # Environment variables template
├── .env                              # Your actual environment variables (not in git)
├── requirements.txt                   # Python dependencies
├── setup.py                          # Environment validation script
├── QUICKSTART.md                      # 5-minute setup guide
├── TROUBLESHOOTING.md                 # Common issues and solutions
└── README.md                          # This file
```

---

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

---

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

---

## 🎬 Demo Examples

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

---

## 💰 Cost Management

### Azure OpenAI Costs
- **text-embedding-3-small**: ~$0.00002 per 1K tokens
- **gpt-4o-mini**: ~$0.00015 per 1K input tokens, ~$0.0006 per 1K output tokens

### Cost Optimization Tips
1. **Batch embedding generation** (included in notebook)
2. **Cache embeddings** to avoid re-processing
3. **Use smaller datasets** for development
4. **Monitor API usage** in Azure Portal
5. **Set up cost alerts** for your subscription

### Sample Costs for Demo
- 50 movies × 200 tokens average = 10K tokens
- Embedding cost: ~$0.0002
- RAG queries: 2-3 queries × ~1K tokens = ~$0.002
- **Total demo cost: < $0.01**

---

## 🔧 Configuration Options

### Vector Dimensions
- **Azure OpenAI text-embedding-3-small**: 1536 dimensions
- **Alternative models**: text-embedding-3-large (3072 dimensions)
- **Custom embeddings**: Adjust VECTOR(n) in DDL scripts

### Performance Tuning
```sql
-- Create additional indexes for better performance
CREATE INDEX idx_movies_popularity ON movies(popularity DESC);
CREATE INDEX idx_movies_genres ON movies(genres);

-- For large datasets, consider partitioning
-- Partition by release year or genre
```

### Batch Processing
```python
# Adjust batch sizes based on your API limits
EMBEDDING_BATCH_SIZE = 10  # Movies per batch
RATE_LIMIT_DELAY = 1       # Seconds between batches
```

---

## 🚨 Security Best Practices

### API Key Management
```bash
# Use environment variables for production
export AZURE_OPENAI_API_KEY="your-key"
export SQL_PASSWORD="your-password"

# Or use Azure Key Vault for enterprise scenarios
```

### Database Security
- Use **Azure AD authentication** when possible
- Enable **SSL/TLS encryption** (already configured in connection string)
- Implement **IP restrictions** on Azure SQL Database
- Use **Managed Identity** for production deployments

### Code Security
- Never commit `config.py` to version control
- Use `.gitignore` to exclude sensitive files
- Implement **retry logic** with exponential backoff
- Add **input validation** for user queries

---

## 📊 Performance Benchmarks

### Vector Search Performance (1536D vectors)
- **10 movies**: ~5ms query time
- **100 movies**: ~15ms query time
- **1,000 movies**: ~50ms query time
- **10,000+ movies**: Consider indexing strategies

### Embedding Generation
- **Single text**: ~200ms per request
- **Batch of 10**: ~1.5s per batch
- **Rate limits**: 3,000 RPM (requests per minute)

### Memory Usage
- **1536D vector**: ~6KB storage per movie
- **1,000 movies**: ~6MB vector storage
- **10,000 movies**: ~60MB vector storage

---

## 🔍 Troubleshooting

### Common Issues

#### 1. Database Connection Errors
```bash
# Check ODBC driver installation
odbcinst -j

# Install ODBC Driver 18 for SQL Server if missing
# Windows: Download from Microsoft
# Linux: sudo apt-get install msodbcsql18
```

#### 2. Azure OpenAI API Errors
```python
# Check your deployment names match
# Verify API key and endpoint
# Ensure sufficient quota

# Test connection:
client.embeddings.create(input="test", model="text-embedding-3-small")
```

#### 3. Vector Dimension Mismatches
```sql
-- Ensure vector dimensions match your embedding model
-- text-embedding-3-small = 1536 dimensions
-- Update DDL if using different models
```

#### 4. Performance Issues
```sql
-- Check if vector indexes exist
SELECT name FROM sys.indexes WHERE object_id = OBJECT_ID('movie_vectors')

-- Recreate if missing
CREATE INDEX idx_movie_vectors_embedding ON movie_vectors(embedding)
```

### Getting Help
- 🚀 [**Quick Start Guide**](QUICKSTART.md) - 5-minute setup
- 🛠️ [**Troubleshooting Guide**](TROUBLESHOOTING.md) - Common issues and solutions
- 📝 Check the [Issues](../../issues) section for common problems
- 📖 Review Azure SQL Database vector documentation
- 🤖 Consult Azure OpenAI service limits and quotas

---

## 🎓 Learning Resources

### Azure Documentation
- [Azure SQL Database Vector Support](https://docs.microsoft.com/azure/azure-sql/database/vector-search)
- [Azure OpenAI Service](https://docs.microsoft.com/azure/cognitive-services/openai/)
- [Vector Search Best Practices](https://docs.microsoft.com/azure/search/vector-search-overview)

### Tutorials and Samples
- [Building RAG Applications with Azure](https://docs.microsoft.com/azure/cognitive-services/openai/tutorials/rag)
- [Vector Databases in Production](https://docs.microsoft.com/azure/architecture/patterns/vector-search)

### Community
- [Azure OpenAI Samples GitHub](https://github.com/Azure-Samples/openai)
- [Azure SQL Database Community](https://techcommunity.microsoft.com/t5/azure-sql-database/ct-p/Azure-SQL-Database)

---

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

## 🙏 Acknowledgments

- **Azure SQL Database Team** for native vector support
- **Azure OpenAI Service** for embedding and generation capabilities
- **The Movies Dataset** contributors on Kaggle
- **Open source community** for Python libraries used

---

## 📞 Support

For questions or issues:
- Create an [Issue](../../issues) in this repository
- Check the [Discussions](../../discussions) for community help
- Review the troubleshooting section above

---

**Happy Vector Searching! 🚀**

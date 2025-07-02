# 🛠️ Troubleshooting Guide - Azure Vector Database Demo

## 🚨 Common Issues and Solutions

### 1. Python Environment Issues

#### Problem: "Python version is too old"
```bash
❌ Python 3.10+ is required. Current version: 3.9.x
```

**Solution:**
```bash
# Windows - Download from python.org
# macOS - Use Homebrew
brew install python@3.11

# Linux - Use package manager
sudo apt update && sudo apt install python3.11
```

#### Problem: "Package not found" errors
```bash
❌ ModuleNotFoundError: No module named 'openai'
```

**Solution:**
```bash
# Ensure you're in the right virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Or install specific package
pip install openai>=1.12.0
```

---

### 2. Azure OpenAI Connection Issues

#### Problem: "Authentication failed"
```bash
❌ Azure OpenAI connection failed: Authentication failed
```

**Solution:**
1. **Check API Key**: Ensure it's copied correctly from Azure Portal
2. **Check Endpoint**: Must end with `.openai.azure.com/`
3. **Check Deployment Names**: Match exactly as created in Azure
4. **Check API Version**: Use `2024-06-01` or later

```python
# Test your connection manually
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_key="your-key",
    api_version="2024-06-01"
)

response = client.embeddings.create(
    input="test",
    model="text-embedding-3-small"  # Your deployment name
)
print(f"Success! Embedding dimensions: {len(response.data[0].embedding)}")
```

#### Problem: "Quota exceeded" or "Rate limit"
```bash
❌ RateLimitError: Rate limit exceeded
```

**Solution:**
```python
# Add retry logic with exponential backoff
import time
import random

def retry_with_backoff(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if "rate" in str(e).lower():
                delay = 2 ** attempt + random.uniform(0, 1)
                print(f"Rate limit hit, waiting {delay:.2f}s...")
                time.sleep(delay)
            else:
                raise e
    raise Exception("Max retries exceeded")
```

#### Problem: "Model deployment not found"
```bash
❌ NotFoundError: The API deployment for this resource does not exist
```

**Solution:**
1. **Check Deployment Names**: Go to Azure OpenAI Studio → Deployments
2. **Verify Model Names**: Ensure you deployed `text-embedding-3-small` and `gpt-4o-mini`
3. **Check Region**: Some models are region-specific

---

### 3. Azure SQL Database Connection Issues

#### Problem: "ODBC Driver not found"
```bash
❌ pyodbc.InterfaceError: ('IM002', '[IM002] [Microsoft][ODBC Driver Manager] Data source name not found')
```

**Solution:**
```bash
# Windows - Download ODBC Driver 18 for SQL Server
# https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server

# Linux
curl https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
curl https://packages.microsoft.com/config/ubuntu/20.04/prod.list | sudo tee /etc/apt/sources.list.d/msprod.list
sudo apt-get update
sudo apt-get install msodbcsql18

# macOS
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew tap microsoft/mssql-release https://github.com/Microsoft/homebrew-mssql-release
brew update
brew install msodbcsql18
```

#### Problem: "Login failed" or "Cannot connect to server"
```bash
❌ pyodbc.InterfaceError: ('28000', "[28000] [Microsoft][ODBC Driver 18 for SQL Server][SQL Server]Login failed")
```

**Solution:**
1. **Check Firewall**: Add your IP to Azure SQL firewall rules
2. **Verify Credentials**: Username/password must be exact
3. **Check Server Name**: Must include `.database.windows.net`
4. **Test with SQL Server Management Studio first**

#### Problem: "SSL/TLS certificate errors"
```bash
❌ SSL Provider: The certificate chain was issued by an authority that is not trusted
```

**Solution:**
```python
# Update connection string with proper SSL settings
connection_string = f"""
    Driver={{ODBC Driver 18 for SQL Server}};
    Server={server};
    Database={database};
    Uid={username};
    Pwd={password};
    Encrypt=yes;
    TrustServerCertificate=yes;  # Use with caution in production
    Connection Timeout=30;
"""
```

#### Problem: "Vector data type not supported"
```bash
❌ Invalid column name 'vector'
```

**Solution:**
1. **Check SQL Server Version**: Vector support requires Azure SQL Database (not SQL Server)
2. **Enable Vector Support**: Contact Azure support if needed
3. **Use Preview Features**: May need to enable preview features in Azure Portal

---

### 4. Dataset and File Issues

#### Problem: "Dataset files not found"
```bash
❌ FileNotFoundError: data/moviesdataset/movies_metadata.csv
```

**Solution:**
```bash
# Create directory structure
mkdir -p data/moviesdataset

# Download dataset from Kaggle
# https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

# Extract to correct location
unzip archive.zip -d data/moviesdataset/
```

#### Problem: "Dataset too large" or "Memory errors"
```bash
❌ MemoryError: Unable to allocate array
```

**Solution:**
```python
# Use chunking for large datasets
import pandas as pd

# Read in smaller chunks
chunk_size = 1000
for chunk in pd.read_csv('movies_metadata.csv', chunksize=chunk_size):
    # Process each chunk
    process_chunk(chunk)

# Or use first N rows for testing
df = pd.read_csv('movies_metadata.csv', nrows=100)
```

---

### 5. Vector and Embedding Issues

#### Problem: "Dimension mismatch"
```bash
❌ Vector dimensions don't match: expected 1536, got 3072
```

**Solution:**
```sql
-- Update table schema to match your embedding model
-- text-embedding-3-small = 1536 dimensions
-- text-embedding-3-large = 3072 dimensions

ALTER TABLE movie_vectors 
ALTER COLUMN embedding VECTOR(3072);  -- Match your model
```

#### Problem: "Vector distance calculations fail"
```bash
❌ Invalid function name 'VECTOR_DISTANCE'
```

**Solution:**
1. **Check Azure SQL Version**: Vector functions require latest version
2. **Update Database**: Ensure you're using Azure SQL Database (not SQL Server)
3. **Alternative Calculation**:
```sql
-- Manual cosine similarity calculation
SELECT 
    title,
    (
        SELECT SUM(a.value * b.value) 
        FROM (SELECT value, ROW_NUMBER() OVER(ORDER BY (SELECT NULL)) as rn FROM VECTOR_TO_ARRAY(@vector1)) a
        JOIN (SELECT value, ROW_NUMBER() OVER(ORDER BY (SELECT NULL)) as rn FROM VECTOR_TO_ARRAY(embedding)) b
        ON a.rn = b.rn
    ) / (
        SQRT((SELECT SUM(POWER(value, 2)) FROM VECTOR_TO_ARRAY(@vector1))) *
        SQRT((SELECT SUM(POWER(value, 2)) FROM VECTOR_TO_ARRAY(embedding)))
    ) as similarity
FROM movie_vectors
```

---

### 6. Performance Issues

#### Problem: "Queries are too slow"
```bash
❌ Query execution time: 45 seconds
```

**Solution:**
```sql
-- Create indexes on frequently queried columns
CREATE INDEX idx_movies_title ON movies(title);
CREATE INDEX idx_movies_genres ON movies(genres);
CREATE INDEX idx_movie_vectors_title ON movie_vectors(title);

-- For vector operations, consider approximate search
-- (when available in future Azure SQL updates)
```

#### Problem: "Memory usage too high"
```python
# Optimize batch processing
BATCH_SIZE = 5  # Reduce from 10 or 20
DELAY_BETWEEN_BATCHES = 2  # Increase delay

# Clear variables when done
del large_dataframe
import gc
gc.collect()
```

---

### 7. Jupyter Notebook Issues

#### Problem: "Kernel keeps dying"
```bash
❌ The kernel appears to have died. It will restart automatically.
```

**Solution:**
```bash
# Increase memory limits
jupyter notebook --NotebookApp.max_buffer_size=1000000000

# Or use JupyterLab
jupyter lab

# Check system resources
# Close other applications
# Use smaller dataset for testing
```

#### Problem: "Modules not found in Jupyter"
```bash
❌ ModuleNotFoundError in notebook but works in terminal
```

**Solution:**
```bash
# Install packages in the Jupyter kernel
python -m ipykernel install --user --name=venv

# Or install directly in notebook
!pip install openai pandas pyodbc

# Restart kernel after installation
```

---

### 8. Cost and Quota Management

#### Problem: "Unexpected charges"
```bash
⚠️ High Azure OpenAI usage detected
```

**Solution:**
```python
# Monitor token usage
import tiktoken

def count_tokens(text, model="text-embedding-3-small"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Set up cost alerts in Azure Portal
# Use smaller datasets for development
# Cache embeddings to avoid re-computation
```

---

### 9. Getting Additional Help

#### Self-Help Resources:
1. **Azure Portal**: Check service health and quotas
2. **Azure Status**: https://status.azure.com/
3. **Documentation**: 
   - [Azure SQL Vector Docs](https://docs.microsoft.com/azure/azure-sql/)
   - [Azure OpenAI Docs](https://docs.microsoft.com/azure/cognitive-services/openai/)

#### Community Support:
1. **GitHub Issues**: [Create an issue](../../issues) in this repository
2. **Stack Overflow**: Tag with `azure-sql-database` and `azure-openai`
3. **Azure Community**: [Microsoft Tech Community](https://techcommunity.microsoft.com/)

#### Professional Support:
1. **Azure Support**: Create support ticket in Azure Portal
2. **Microsoft FastTrack**: For enterprise scenarios
3. **Azure Architecture Center**: For design guidance

---

### 10. Verification Checklist

Before reporting issues, verify:

- [ ] Python 3.10+ installed
- [ ] All packages in requirements.txt installed
- [ ] config.py created and filled out correctly
- [ ] Azure OpenAI deployments created and accessible
- [ ] Azure SQL Database created with vector support
- [ ] Firewall rules configured for database access
- [ ] Dataset files downloaded and placed correctly
- [ ] setup.py validation script passes
- [ ] Network connectivity to Azure services

#### Run This Quick Diagnostic:
```bash
python setup.py  # Should show all green checkmarks ✅
```

If you're still experiencing issues after checking this guide, please create a [GitHub issue](../../issues) with:
- Error message (full stack trace)
- Operating system and Python version
- Output of `python setup.py`
- Steps to reproduce the issue

**We're here to help! 🤝**

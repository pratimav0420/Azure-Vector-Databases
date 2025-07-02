# Azure OpenAI Configuration for Movie Vector Database Demo
# Copy this file to config.py and update with your actual values

# Azure OpenAI Service Configuration
AZURE_OPENAI_ENDPOINT = "https://your-openai-resource.openai.azure.com/"
AZURE_OPENAI_API_KEY = "your-api-key-here"  # Use environment variable in production: os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = "2024-06-01"

# Model Configuration
EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dimensions
GENERATION_MODEL = "gpt-4o-mini"

# Azure SQL Database Configuration
# Update these with your actual Azure SQL Database details
SERVER = "your-server.database.windows.net"
DATABASE = "your-database-name"
USERNAME = "your-username"
PASSWORD = "your-password"  # Use environment variable in production: os.getenv("SQL_PASSWORD")

# Performance and Cost Configuration
EMBEDDING_BATCH_SIZE = 10  # Process embeddings in batches to manage rate limits
MAX_DEMO_MOVIES = 50  # Limit movies for demo to control costs
RATE_LIMIT_DELAY = 1  # Seconds to wait between API calls

# Vector Search Configuration
DEFAULT_TOP_K = 5  # Default number of similar items to retrieve
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity score for relevant results

# RAG Configuration
MAX_CONTEXT_TOKENS = 3000  # Maximum tokens to send to GPT model
GENERATION_MAX_TOKENS = 1000  # Maximum tokens for generated response
GENERATION_TEMPERATURE = 0.7  # Creativity level for generation

# Production Security Notes:
# 1. Store API keys in Azure Key Vault or environment variables
# 2. Use Managed Identity for Azure SQL Database authentication
# 3. Implement proper retry logic with exponential backoff
# 4. Add request logging and monitoring
# 5. Set up cost alerts for Azure OpenAI usage

"""
MongoDB vCore - Advanced Vector Operations
Equivalent to Advanced_Vector_Operations.sql

This script demonstrates advanced vector database operations and analytics
using Azure Cosmos DB for MongoDB vCore
"""

from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Load environment variables
load_dotenv()

# Configuration
MONGODB_CONNECTION_STRING = os.getenv("MONGODB_CONNECTION_STRING")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

DATABASE_NAME = "AdvancedVectorDB"
COLLECTION_NAME = "advanced_vectors"

# Validate required environment variables
required_vars = ["MONGODB_CONNECTION_STRING", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize clients
mongo_client = MongoClient(MONGODB_CONNECTION_STRING)
openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION
)

# Get database and collection
database = mongo_client[DATABASE_NAME]
collection = database[COLLECTION_NAME]

print(f"Database '{DATABASE_NAME}' ready")
print(f"Collection '{COLLECTION_NAME}' ready")
print(f"Using embedding model: {EMBEDDING_MODEL}")

# Clear existing data for fresh demo
collection.delete_many({})
print("Cleared existing data")

# Function to get embedding from Azure OpenAI
def get_embedding(text, model=EMBEDDING_MODEL):
    """Get embedding from Azure OpenAI"""
    try:
        response = openai_client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

# Create comprehensive vector search index
vector_index_model = SearchIndexModel(
    definition={
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 1536,
                "similarity": "cosine"
            },
            {
                "type": "filter",
                "path": "category"
            },
            {
                "type": "filter", 
                "path": "year"
            },
            {
                "type": "filter",
                "path": "rating"
            }
        ]
    },
    name="advanced_vector_index"
)

try:
    # Check if index exists
    existing_indexes = list(collection.list_search_indexes())
    index_exists = any(idx.get('name') == 'advanced_vector_index' for idx in existing_indexes)
    
    if not index_exists:
        collection.create_search_index(vector_index_model)
        print("✅ Created advanced vector search index")
        print("⏳ Waiting for index to be ready...")
        time.sleep(30)
    else:
        print("✅ Advanced vector search index already exists")

except Exception as e:
    print(f"⚠️ Vector index creation: {e}")

# Extended dataset for advanced operations
advanced_dataset = [
    # Movies
    {"text": "Epic space opera with lightsabers and the Force", "category": "movie", "subcategory": "sci-fi", "year": 1977, "rating": 8.6},
    {"text": "Mob family saga with Don Corleone", "category": "movie", "subcategory": "crime", "year": 1972, "rating": 9.2},
    {"text": "Dream heist thriller with multiple layers of reality", "category": "movie", "subcategory": "sci-fi", "year": 2010, "rating": 8.7},
    {"text": "Boxer's journey through different timelines", "category": "movie", "subcategory": "crime", "year": 1994, "rating": 8.9},
    {"text": "Shark attacks beach town in summer blockbuster", "category": "movie", "subcategory": "thriller", "year": 1975, "rating": 8.0},
    
    # Books
    {"text": "Dystopian future with Big Brother watching", "category": "book", "subcategory": "dystopian", "year": 1949, "rating": 9.0},
    {"text": "Hobbit's adventure to destroy the ring", "category": "book", "subcategory": "fantasy", "year": 1954, "rating": 9.1},
    {"text": "Mockingbird symbolism in Southern legal drama", "category": "book", "subcategory": "drama", "year": 1960, "rating": 8.8},
    {"text": "Green light across the bay in Jazz Age", "category": "book", "subcategory": "classic", "year": 1925, "rating": 8.5},
    
    # TV Shows
    {"text": "Chemistry teacher turns drug manufacturer", "category": "tv", "subcategory": "drama", "year": 2008, "rating": 9.4},
    {"text": "Medieval fantasy with dragons and throne politics", "category": "tv", "subcategory": "fantasy", "year": 2011, "rating": 8.7},
    {"text": "Mafia family therapy sessions", "category": "tv", "subcategory": "crime", "year": 1999, "rating": 9.2},
    {"text": "Six friends living in New York apartment", "category": "tv", "subcategory": "comedy", "year": 1994, "rating": 8.9},
    
    # Music
    {"text": "Bohemian opera rock with piano ballad sections", "category": "music", "subcategory": "rock", "year": 1975, "rating": 9.5},
    {"text": "Thriller album with moonwalk dance moves", "category": "music", "subcategory": "pop", "year": 1982, "rating": 9.3},
    {"text": "Hotel California guitar solos and mysterious lyrics", "category": "music", "subcategory": "rock", "year": 1976, "rating": 9.1},
    {"text": "Purple rain soundtrack with guitar mastery", "category": "music", "subcategory": "funk", "year": 1984, "rating": 8.9},
    
    # Games
    {"text": "Plumber jumping on mushrooms in castle rescue", "category": "game", "subcategory": "platform", "year": 1985, "rating": 8.7},
    {"text": "Block puzzle game falling from the sky", "category": "game", "subcategory": "puzzle", "year": 1984, "rating": 8.5},
    {"text": "Post-apocalyptic wasteland with bottle caps currency", "category": "game", "subcategory": "rpg", "year": 1997, "rating": 9.1},
    {"text": "Medieval fantasy dragons and magic spells", "category": "game", "subcategory": "rpg", "year": 2011, "rating": 9.3}
]

print(f"\n📚 Processing {len(advanced_dataset)} items for advanced vector operations...")

# Generate embeddings and store documents
documents_to_insert = []
successful_embeddings = 0

for item in advanced_dataset:
    print(f"Processing: {item['text'][:50]}...")
    
    embedding = get_embedding(item['text'])
    
    if embedding:
        document = {
            "text": item["text"],
            "category": item["category"],
            "subcategory": item["subcategory"],
            "year": item["year"],
            "rating": item["rating"],
            "embedding": embedding,
            "embedding_model": EMBEDDING_MODEL,
            "created_at": time.time()
        }
        documents_to_insert.append(document)
        successful_embeddings += 1
        
        # Rate limiting
        time.sleep(0.3)
    else:
        print(f"⚠️ Failed to generate embedding")

# Bulk insert documents
if documents_to_insert:
    try:
        result = collection.insert_many(documents_to_insert)
        print(f"✅ Successfully stored {len(result.inserted_ids)} documents")
    except Exception as e:
        print(f"❌ Error during bulk insert: {e}")

print(f"📊 Successfully processed {successful_embeddings} out of {len(advanced_dataset)} items")

# Advanced Operation 1: Multi-Vector Search with Filters
print("\n🎯 **Advanced Operation 1: Multi-Vector Search with Filters**")

def filtered_vector_search(query_text, category_filter=None, year_range=None, rating_min=None, top_k=5):
    """Advanced vector search with multiple filters"""
    
    query_embedding = get_embedding(query_text)
    if not query_embedding:
        return []
    
    try:
        # Build filter conditions
        filters = {}
        if category_filter:
            filters["category"] = {"$eq": category_filter}
        if year_range:
            filters["year"] = {"$gte": year_range[0], "$lte": year_range[1]}
        if rating_min:
            filters["rating"] = {"$gte": rating_min}
        
        # Vector search with filters
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "advanced_vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": top_k,
                    "filter": filters if filters else {}
                }
            },
            {
                "$project": {
                    "text": 1,
                    "category": 1,
                    "subcategory": 1,
                    "year": 1,
                    "rating": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        results = list(collection.aggregate(pipeline))
        return results
        
    except Exception as e:
        print(f"Error in filtered search: {e}")
        return []

# Test filtered searches
print("🔍 Filtered Vector Searches:")

# Search 1: Only movies about crime
query = "mafia organized crime family"
results = filtered_vector_search(query, category_filter="movie", rating_min=8.0)
print(f"\nQuery: '{query}' (movies only, rating >= 8.0)")
for result in results:
    print(f"  • {result['text'][:50]}... ({result['category']}, {result['year']}, {result['rating']}) - Score: {result.get('score', 0):.4f}")

# Search 2: High-rated content from 1970s-1980s
query = "epic adventure story"
results = filtered_vector_search(query, year_range=(1970, 1989), rating_min=8.5)
print(f"\nQuery: '{query}' (1970s-1980s, rating >= 8.5)")
for result in results:
    print(f"  • {result['text'][:50]}... ({result['category']}, {result['year']}, {result['rating']}) - Score: {result.get('score', 0):.4f}")

# Advanced Operation 2: Vector Clustering Analysis
print("\n🎯 **Advanced Operation 2: Vector Clustering Analysis**")

def perform_vector_clustering(n_clusters=5):
    """Perform K-means clustering on embeddings"""
    
    # Get all documents with embeddings
    docs = list(collection.find({"embedding": {"$exists": True}}))
    
    if len(docs) < n_clusters:
        print(f"Not enough documents ({len(docs)}) for {n_clusters} clusters")
        return
    
    # Extract embeddings and metadata
    embeddings = [doc['embedding'] for doc in docs]
    metadata = [{
        'text': doc['text'][:50] + '...',
        'category': doc['category'],
        'year': doc['year']
    } for doc in docs]
    
    # Perform K-means clustering
    embeddings_array = np.array(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_array)
    
    # Analyze clusters
    print(f"📊 K-means clustering results ({n_clusters} clusters):")
    
    for cluster_id in range(n_clusters):
        cluster_docs = [metadata[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
        print(f"\nCluster {cluster_id} ({len(cluster_docs)} items):")
        
        # Show category distribution
        categories = [doc['category'] for doc in cluster_docs]
        category_counts = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        print(f"  Categories: {dict(category_counts)}")
        
        # Show sample items
        for doc in cluster_docs[:3]:
            print(f"    • {doc['text']} ({doc['category']}, {doc['year']})")
    
    return cluster_labels, embeddings_array

# Perform clustering
cluster_labels, embeddings_array = perform_vector_clustering(n_clusters=4)

# Advanced Operation 3: Similarity Matrix Analysis
print("\n🎯 **Advanced Operation 3: Similarity Matrix Analysis**")

def analyze_similarity_patterns():
    """Analyze similarity patterns across different categories"""
    
    # Get documents by category
    categories = ["movie", "book", "tv", "music", "game"]
    category_docs = {}
    
    for category in categories:
        docs = list(collection.find({"category": category, "embedding": {"$exists": True}}).limit(4))
        if docs:
            category_docs[category] = docs
    
    if len(category_docs) < 2:
        print("Not enough categories for similarity analysis")
        return
    
    print("📈 Cross-category similarity analysis:")
    
    # Calculate average similarities between categories
    for cat1 in category_docs:
        for cat2 in category_docs:
            if cat1 <= cat2:  # Avoid duplicate comparisons
                continue
            
            similarities = []
            for doc1 in category_docs[cat1]:
                for doc2 in category_docs[cat2]:
                    emb1 = np.array(doc1['embedding'])
                    emb2 = np.array(doc2['embedding'])
                    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    similarities.append(similarity)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                print(f"  {cat1} ↔ {cat2}: {avg_similarity:.4f}")

# Perform similarity analysis
analyze_similarity_patterns()

# Advanced Operation 4: Temporal Vector Analysis
print("\n🎯 **Advanced Operation 4: Temporal Vector Analysis**")

def temporal_vector_analysis():
    """Analyze how vector similarities change over time periods"""
    
    # Group documents by decade
    decades = {}
    all_docs = list(collection.find({"embedding": {"$exists": True}}))
    
    for doc in all_docs:
        decade = (doc['year'] // 10) * 10
        if decade not in decades:
            decades[decade] = []
        decades[decade].append(doc)
    
    print("📅 Temporal similarity analysis:")
    
    # Compare decades
    decade_list = sorted(decades.keys())
    for i, decade1 in enumerate(decade_list):
        for decade2 in decade_list[i+1:]:
            similarities = []
            
            for doc1 in decades[decade1]:
                for doc2 in decades[decade2]:
                    emb1 = np.array(doc1['embedding'])
                    emb2 = np.array(doc2['embedding'])
                    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    similarities.append(similarity)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                print(f"  {decade1}s ↔ {decade2}s: {avg_similarity:.4f} ({len(similarities)} comparisons)")

# Perform temporal analysis
temporal_vector_analysis()

# Advanced Operation 5: Hybrid Search (Vector + Text)
print("\n🎯 **Advanced Operation 5: Hybrid Search (Vector + Text)**")

def hybrid_search(query_text, text_keywords=None, top_k=5):
    """Combine vector search with traditional text search"""
    
    query_embedding = get_embedding(query_text)
    if not query_embedding:
        return []
    
    try:
        # Build text filter if keywords provided
        text_filter = {}
        if text_keywords:
            # Create regex pattern for keywords
            keyword_patterns = [{"text": {"$regex": kw, "$options": "i"}} for kw in text_keywords]
            text_filter = {"$or": keyword_patterns}
        
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "advanced_vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": top_k * 2,
                    "filter": text_filter if text_filter else {}
                }
            },
            {
                "$project": {
                    "text": 1,
                    "category": 1,
                    "year": 1,
                    "rating": 1,
                    "vector_score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        results = list(collection.aggregate(pipeline))
        
        # Re-rank by combining vector score with text relevance
        if text_keywords:
            for result in results:
                text_matches = sum(1 for kw in text_keywords 
                                 if kw.lower() in result['text'].lower())
                text_score = text_matches / len(text_keywords)
                # Combine scores (weighted)
                result['hybrid_score'] = 0.7 * result['vector_score'] + 0.3 * text_score
            
            results.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
        
        return results[:top_k]
        
    except Exception as e:
        print(f"Error in hybrid search: {e}")
        return []

# Test hybrid search
print("🔍 Hybrid Search Examples:")

query = "epic adventure story"
keywords = ["fantasy", "dragon", "magic"]
results = hybrid_search(query, text_keywords=keywords, top_k=3)
print(f"\nQuery: '{query}' with keywords: {keywords}")
for result in results:
    vector_score = result.get('vector_score', 0)
    hybrid_score = result.get('hybrid_score', vector_score)
    print(f"  • {result['text'][:60]}... (Vector: {vector_score:.4f}, Hybrid: {hybrid_score:.4f})")

# Advanced Operation 6: Performance Benchmarking
print("\n🎯 **Advanced Operation 6: Performance Benchmarking**")

def benchmark_vector_operations():
    """Benchmark different vector operation performance"""
    
    print("⏱️ Vector Operations Performance:")
    
    # Test 1: Simple vector search
    query_embedding = get_embedding("action adventure movie")
    
    start_time = time.time()
    pipeline = [
        {"$vectorSearch": {
            "index": "advanced_vector_index",
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": 50,
            "limit": 10
        }},
        {"$project": {"text": 1, "score": {"$meta": "vectorSearchScore"}}}
    ]
    results = list(collection.aggregate(pipeline))
    simple_search_time = time.time() - start_time
    
    print(f"  Simple vector search: {simple_search_time*1000:.2f} ms ({len(results)} results)")
    
    # Test 2: Filtered vector search
    start_time = time.time()
    pipeline = [
        {"$vectorSearch": {
            "index": "advanced_vector_index",
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": 50,
            "limit": 10,
            "filter": {"rating": {"$gte": 8.0}}
        }},
        {"$project": {"text": 1, "rating": 1, "score": {"$meta": "vectorSearchScore"}}}
    ]
    results = list(collection.aggregate(pipeline))
    filtered_search_time = time.time() - start_time
    
    print(f"  Filtered vector search: {filtered_search_time*1000:.2f} ms ({len(results)} results)")
    
    # Test 3: Aggregation with vector search
    start_time = time.time()
    pipeline = [
        {"$vectorSearch": {
            "index": "advanced_vector_index",
            "path": "embedding", 
            "queryVector": query_embedding,
            "numCandidates": 100,
            "limit": 20
        }},
        {"$group": {
            "_id": "$category",
            "count": {"$sum": 1},
            "avg_rating": {"$avg": "$rating"},
            "avg_score": {"$avg": {"$meta": "vectorSearchScore"}}
        }},
        {"$sort": {"avg_score": -1}}
    ]
    results = list(collection.aggregate(pipeline))
    aggregation_time = time.time() - start_time
    
    print(f"  Vector search + aggregation: {aggregation_time*1000:.2f} ms ({len(results)} groups)")
    
    # Display aggregation results
    print("    Category breakdown:")
    for result in results:
        print(f"      {result['_id']}: {result['count']} items, "
              f"avg rating: {result['avg_rating']:.1f}, "
              f"avg score: {result.get('avg_score', 0):.4f}")

# Run benchmarks
benchmark_vector_operations()

# Collection and Index Statistics
print("\n📊 **Collection and Index Statistics**")

def get_collection_stats():
    """Get detailed collection and index statistics"""
    
    # Collection stats
    try:
        stats = database.command("collStats", COLLECTION_NAME)
        print(f"Collection Statistics:")
        print(f"  Document count: {stats.get('count', 0)}")
        print(f"  Data size: {stats.get('size', 0)} bytes")
        print(f"  Average document size: {stats.get('avgObjSize', 0)} bytes")
        print(f"  Index count: {stats.get('nindexes', 0)}")
        print(f"  Total index size: {stats.get('totalIndexSize', 0)} bytes")
    except Exception as e:
        print(f"Could not get collection stats: {e}")
    
    # Index information
    try:
        indexes = list(collection.list_search_indexes())
        print(f"\nSearch Index Information:")
        for idx in indexes:
            print(f"  Index: {idx.get('name')}")
            print(f"  Status: {idx.get('status')}")
            if 'definition' in idx:
                fields = idx['definition'].get('fields', [])
                vector_fields = [f for f in fields if f.get('type') == 'vector']
                filter_fields = [f for f in fields if f.get('type') == 'filter']
                print(f"  Vector fields: {len(vector_fields)}")
                print(f"  Filter fields: {len(filter_fields)}")
    except Exception as e:
        print(f"Could not get index information: {e}")

# Get statistics
get_collection_stats()

print("\n✅ Advanced vector operations demo completed!")
print("\nOperations demonstrated:")
print("• Multi-dimensional vector search with filters")
print("• K-means clustering on high-dimensional embeddings")
print("• Cross-category similarity matrix analysis")
print("• Temporal vector pattern analysis")
print("• Hybrid search combining vector and text")
print("• Performance benchmarking of vector operations")
print("• Advanced aggregation pipelines with vector search")
print("• Collection and index statistics analysis")

print("\n🚀 Production considerations:")
print("• Vector index optimization for large datasets")
print("• Sharding strategies for vector collections")
print("• Embedding model versioning and updates")
print("• Real-time vs batch vector processing")
print("• Cost optimization for vector operations")
print("• Monitoring and alerting for vector performance")

# Cleanup (uncomment to clean up test data)
# collection.delete_many({})
# print("🧹 Cleaned up test data")

# Close connection
mongo_client.close()

# ========================================
# Azure Cosmos DB NoSQL API Advanced Vector Operations
# Movie Dataset Vector Database Demo
# ========================================

"""
Advanced Vector Operations for Azure Cosmos DB NoSQL API

This file contains advanced vector operations and queries that demonstrate
the sophisticated capabilities of Cosmos DB for vector database applications.

Equivalent to Advanced_Vector_Operations.sql for Azure SQL Database.
"""

from azure.cosmos import CosmosClient, PartitionKey
from openai import AzureOpenAI
import os
import numpy as np
import time
from dotenv import load_dotenv
from typing import List, Dict, Any
import json

# Load environment variables
load_dotenv()

# Configuration
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

DATABASE_NAME = "AdvancedVectorDB"
CONTAINER_NAME = "advanced_vectors"

class CosmosVectorOperations:
    """Advanced vector operations for Cosmos DB"""
    
    def __init__(self):
        self.cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
        self.openai_client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION
        )
        self.database = None
        self.container = None
        self.setup_database()
    
    def setup_database(self):
        """Setup database and container with advanced vector capabilities"""
        
        # Create database
        self.database = self.cosmos_client.create_database_if_not_exists(id=DATABASE_NAME)
        print(f"✅ Database '{DATABASE_NAME}' ready")
        
        # Advanced vector embedding policy with multiple vector paths
        vector_embedding_policy = {
            "vectorEmbeddings": [
                {
                    "path": "/content_embedding",
                    "dataType": "float32",
                    "distanceFunction": "cosine",
                    "dimensions": 1536
                },
                {
                    "path": "/title_embedding",
                    "dataType": "float32", 
                    "distanceFunction": "cosine",
                    "dimensions": 1536
                },
                {
                    "path": "/genre_embedding",
                    "dataType": "float32",
                    "distanceFunction": "euclidean",
                    "dimensions": 1536
                }
            ]
        }
        
        # Advanced indexing policy
        indexing_policy = {
            "indexingMode": "consistent",
            "automatic": True,
            "includedPaths": [{"path": "/*"}],
            "excludedPaths": [
                {"path": "/content_embedding/*"},
                {"path": "/title_embedding/*"},
                {"path": "/genre_embedding/*"}
            ],
            "vectorIndexes": [
                {
                    "path": "/content_embedding",
                    "type": "quantizedFlat"
                },
                {
                    "path": "/title_embedding", 
                    "type": "quantizedFlat"
                },
                {
                    "path": "/genre_embedding",
                    "type": "quantizedFlat"
                }
            ]
        }
        
        # Create container with advanced vector capabilities
        self.container = self.database.create_container_if_not_exists(
            id=CONTAINER_NAME,
            partition_key=PartitionKey(path="/document_type"),
            indexing_policy=indexing_policy,
            vector_embedding_policy=vector_embedding_policy,
            offer_throughput=1000
        )
        print(f"✅ Container '{CONTAINER_NAME}' ready with advanced vector capabilities")
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding from Azure OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=EMBEDDING_MODEL
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Error getting embedding: {e}")
            return None
    
    def create_sample_documents(self):
        """Create sample documents with multiple vector embeddings"""
        
        print("📝 Creating sample documents with multiple embeddings...")
        
        sample_movies = [
            {
                "title": "The Matrix",
                "content": "A computer programmer discovers reality is a simulation and joins a rebellion against machines",
                "genres": "Action Sci-Fi Thriller",
                "year": 1999,
                "rating": 8.7,
                "director": "The Wachowskis"
            },
            {
                "title": "Inception", 
                "content": "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea",
                "genres": "Action Sci-Fi Thriller",
                "year": 2010,
                "rating": 8.8,
                "director": "Christopher Nolan"
            },
            {
                "title": "The Godfather",
                "content": "The aging patriarch of an organized crime dynasty transfers control to his reluctant son",
                "genres": "Crime Drama",
                "year": 1972,
                "rating": 9.2,
                "director": "Francis Ford Coppola"
            },
            {
                "title": "Pulp Fiction",
                "content": "The lives of two mob hitmen, a boxer, and a gangster intertwine in four tales of violence and redemption",
                "genres": "Crime Drama Thriller",
                "year": 1994,
                "rating": 8.9,
                "director": "Quentin Tarantino"
            },
            {
                "title": "Toy Story",
                "content": "A cowboy doll is profoundly threatened when a new spaceman figure supplants him as top toy in a boy's room",
                "genres": "Animation Family Comedy",
                "year": 1995,
                "rating": 8.3,
                "director": "John Lasseter"
            }
        ]
        
        for i, movie in enumerate(sample_movies, 1):
            print(f"Processing movie {i}/{len(sample_movies)}: {movie['title']}")
            
            # Generate embeddings for different aspects
            title_embedding = self.get_embedding(movie['title'])
            content_embedding = self.get_embedding(movie['content'])
            genre_embedding = self.get_embedding(movie['genres'])
            
            if all([title_embedding, content_embedding, genre_embedding]):
                doc = {
                    "id": f"movie_{i}",
                    "document_type": "movie",
                    "title": movie['title'],
                    "content": movie['content'],
                    "genres": movie['genres'],
                    "year": movie['year'],
                    "rating": movie['rating'],
                    "director": movie['director'],
                    "title_embedding": title_embedding,
                    "content_embedding": content_embedding,
                    "genre_embedding": genre_embedding,
                    "created_at": time.strftime('%Y-%m-%dT%H:%M:%SZ')
                }
                
                try:
                    self.container.create_item(doc)
                    print(f"  ✅ Stored with 3 embeddings")
                except Exception as e:
                    print(f"  ❌ Error storing: {e}")
            else:
                print(f"  ❌ Failed to get embeddings")
            
            time.sleep(0.5)  # Rate limiting
        
        print("✅ Sample documents created")
    
    def multi_vector_search(self, query: str, search_type: str = "content", top_k: int = 3):
        """Perform search using different vector embeddings"""
        
        print(f"🔍 Multi-vector search: '{query}' (type: {search_type})")
        
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []
        
        # Choose embedding path based on search type
        embedding_paths = {
            "content": "content_embedding",
            "title": "title_embedding", 
            "genre": "genre_embedding"
        }
        
        embedding_path = embedding_paths.get(search_type, "content_embedding")
        
        search_query = f"""
        SELECT TOP @topK
            c.title,
            c.content,
            c.genres,
            c.rating,
            c.director,
            c.year,
            VectorDistance('cosine', c.{embedding_path}, @queryEmbedding) as distance,
            1 - VectorDistance('cosine', c.{embedding_path}, @queryEmbedding) as similarity
        FROM c 
        WHERE c.document_type = 'movie' AND IS_DEFINED(c.{embedding_path})
        ORDER BY VectorDistance('cosine', c.{embedding_path}, @queryEmbedding)
        """
        
        results = list(self.container.query_items(
            query=search_query,
            parameters=[
                {"name": "@queryEmbedding", "value": query_embedding},
                {"name": "@topK", "value": top_k}
            ],
            enable_cross_partition_query=True
        ))
        
        print(f"📚 Found {len(results)} results using {search_type} embeddings:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['title']} ({result['year']}) - Similarity: {result['similarity']:.4f}")
            print(f"     Director: {result['director']}, Rating: {result['rating']}")
        
        return results
    
    def hybrid_vector_search(self, query: str, weights: Dict[str, float] = None, top_k: int = 3):
        """Perform hybrid search combining multiple vector embeddings"""
        
        if weights is None:
            weights = {"content": 0.5, "title": 0.3, "genre": 0.2}
        
        print(f"🔍 Hybrid vector search: '{query}'")
        print(f"📊 Weights: {weights}")
        
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []
        
        # Note: Cosmos DB doesn't support weighted vector operations directly in SQL
        # This is a simulation - in practice, you'd retrieve results and combine scores in application logic
        
        all_results = []
        
        # Get results from each embedding type
        for search_type, weight in weights.items():
            if weight > 0:
                results = self.multi_vector_search(query, search_type, top_k * 2)
                for result in results:
                    result['weighted_similarity'] = result['similarity'] * weight
                    result['search_type'] = search_type
                all_results.extend(results)
        
        # Combine and rank results (simplified aggregation)
        movie_scores = {}
        for result in all_results:
            movie_id = result['title']
            if movie_id not in movie_scores:
                movie_scores[movie_id] = {
                    'movie': result,
                    'total_score': 0,
                    'score_breakdown': {}
                }
            
            movie_scores[movie_id]['total_score'] += result['weighted_similarity']
            movie_scores[movie_id]['score_breakdown'][result['search_type']] = result['weighted_similarity']
        
        # Sort by total score
        sorted_results = sorted(
            movie_scores.values(),
            key=lambda x: x['total_score'],
            reverse=True
        )[:top_k]
        
        print(f"🎯 Hybrid results (top {top_k}):")
        for i, item in enumerate(sorted_results, 1):
            movie = item['movie']
            print(f"  {i}. {movie['title']} - Combined Score: {item['total_score']:.4f}")
            print(f"     Breakdown: {item['score_breakdown']}")
        
        return sorted_results
    
    def vector_clustering_analysis(self):
        """Analyze vector clustering patterns"""
        
        print("📊 Vector Clustering Analysis")
        
        # Get all movies with their embeddings
        all_movies_query = """
        SELECT 
            c.title,
            c.genres,
            c.year,
            c.rating,
            c.content_embedding
        FROM c 
        WHERE c.document_type = 'movie' AND IS_DEFINED(c.content_embedding)
        """
        
        movies = list(self.container.query_items(query=all_movies_query, enable_cross_partition_query=True))
        
        if len(movies) < 2:
            print("❌ Need at least 2 movies for clustering analysis")
            return
        
        print(f"📚 Analyzing {len(movies)} movies")
        
        # Calculate pairwise similarities
        similarities = []
        for i, movie1 in enumerate(movies):
            for j, movie2 in enumerate(movies[i+1:], i+1):
                
                # Calculate similarity using Cosmos DB query
                similarity_query = """
                SELECT 
                    @movie1Title as movie1,
                    @movie2Title as movie2,
                    1 - VectorDistance('cosine', @embedding1, @embedding2) as similarity
                FROM c 
                WHERE c.id = '1'  -- Dummy condition
                """
                
                sim_result = list(self.container.query_items(
                    query=similarity_query,
                    parameters=[
                        {"name": "@movie1Title", "value": movie1['title']},
                        {"name": "@movie2Title", "value": movie2['title']},
                        {"name": "@embedding1", "value": movie1['content_embedding']},
                        {"name": "@embedding2", "value": movie2['content_embedding']}
                    ],
                    enable_cross_partition_query=True
                ))
                
                if sim_result:
                    similarity = sim_result[0]['similarity']
                    similarities.append({
                        'movie1': movie1['title'],
                        'movie2': movie2['title'],
                        'similarity': similarity,
                        'genre1': movie1['genres'],
                        'genre2': movie2['genres']
                    })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        print("🎯 Most Similar Movie Pairs:")
        for i, sim in enumerate(similarities[:5], 1):
            print(f"  {i}. {sim['movie1']} ↔ {sim['movie2']}")
            print(f"     Similarity: {sim['similarity']:.4f}")
            print(f"     Genres: {sim['genre1']} | {sim['genre2']}")
            print()
        
        print("🎯 Least Similar Movie Pairs:")
        for i, sim in enumerate(similarities[-3:], 1):
            print(f"  {i}. {sim['movie1']} ↔ {sim['movie2']}")
            print(f"     Similarity: {sim['similarity']:.4f}")
            print(f"     Genres: {sim['genre1']} | {sim['genre2']}")
            print()
    
    def vector_aggregation_queries(self):
        """Demonstrate vector aggregation operations"""
        
        print("📊 Vector Aggregation Operations")
        
        # Average rating by genre clusters
        print("\n1️⃣ Finding average ratings for similar content clusters")
        
        # For each movie, find its most similar movie and compare ratings
        comparison_query = """
        SELECT 
            c1.title as movie1,
            c1.rating as rating1,
            c1.genres as genres1
        FROM c as c1
        WHERE c1.document_type = 'movie'
        """
        
        movies = list(self.container.query_items(query=comparison_query, enable_cross_partition_query=True))
        
        for movie in movies[:3]:  # Limit for demo
            print(f"\n🎬 Analyzing: {movie['movie1']}")
            
            # Find most similar movies
            similar_query = """
            SELECT TOP 3
                c.title,
                c.rating,
                c.genres,
                1 - VectorDistance('cosine', c.content_embedding, @targetEmbedding) as similarity
            FROM c 
            WHERE c.document_type = 'movie' 
                AND c.title != @targetTitle
                AND IS_DEFINED(c.content_embedding)
            ORDER BY VectorDistance('cosine', c.content_embedding, @targetEmbedding)
            """
            
            # Get target movie's embedding
            target_embedding_query = """
            SELECT c.content_embedding 
            FROM c 
            WHERE c.title = @targetTitle AND c.document_type = 'movie'
            """
            
            target_emb_result = list(self.container.query_items(
                query=target_embedding_query,
                parameters=[{"name": "@targetTitle", "value": movie['movie1']}],
                enable_cross_partition_query=True
            ))
            
            if target_emb_result:
                target_embedding = target_emb_result[0]['content_embedding']
                
                similar_results = list(self.container.query_items(
                    query=similar_query,
                    parameters=[
                        {"name": "@targetEmbedding", "value": target_embedding},
                        {"name": "@targetTitle", "value": movie['movie1']}
                    ],
                    enable_cross_partition_query=True
                ))
                
                if similar_results:
                    avg_rating = sum(r['rating'] for r in similar_results) / len(similar_results)
                    print(f"  Similar movies average rating: {avg_rating:.2f}")
                    print(f"  Target movie rating: {movie['rating1']}")
                    print(f"  Similar movies:")
                    for sim in similar_results:
                        print(f"    - {sim['title']} (Rating: {sim['rating']}, Similarity: {sim['similarity']:.3f})")
    
    def vector_performance_benchmarks(self):
        """Run performance benchmarks on vector operations"""
        
        print("⚡ Vector Performance Benchmarks")
        
        # Test 1: Simple vector similarity query
        print("\n1️⃣ Simple Vector Similarity Query")
        
        # Get a sample embedding
        sample_query = """
        SELECT TOP 1 c.content_embedding 
        FROM c 
        WHERE c.document_type = 'movie' AND IS_DEFINED(c.content_embedding)
        """
        
        sample_result = list(self.container.query_items(query=sample_query, enable_cross_partition_query=True))
        
        if sample_result:
            test_embedding = sample_result[0]['content_embedding']
            
            # Benchmark query
            start_time = time.time()
            
            perf_query = """
            SELECT TOP 5
                c.title,
                VectorDistance('cosine', c.content_embedding, @testEmbedding) as distance
            FROM c 
            WHERE c.document_type = 'movie' AND IS_DEFINED(c.content_embedding)
            ORDER BY VectorDistance('cosine', c.content_embedding, @testEmbedding)
            """
            
            results = list(self.container.query_items(
                query=perf_query,
                parameters=[{"name": "@testEmbedding", "value": test_embedding}],
                enable_cross_partition_query=True
            ))
            
            end_time = time.time()
            
            print(f"  Query time: {(end_time - start_time)*1000:.2f} ms")
            print(f"  Results returned: {len(results)}")
            print(f"  Top result: {results[0]['title'] if results else 'None'}")
        
        # Test 2: Complex vector query with filters
        print("\n2️⃣ Complex Vector Query with Filters")
        
        if sample_result:
            start_time = time.time()
            
            complex_query = """
            SELECT TOP 3
                c.title,
                c.rating,
                c.year,
                VectorDistance('cosine', c.content_embedding, @testEmbedding) as distance
            FROM c 
            WHERE c.document_type = 'movie' 
                AND IS_DEFINED(c.content_embedding)
                AND c.rating >= 8.0
                AND c.year >= 1990
            ORDER BY VectorDistance('cosine', c.content_embedding, @testEmbedding)
            """
            
            complex_results = list(self.container.query_items(
                query=complex_query,
                parameters=[{"name": "@testEmbedding", "value": test_embedding}],
                enable_cross_partition_query=True
            ))
            
            end_time = time.time()
            
            print(f"  Query time: {(end_time - start_time)*1000:.2f} ms")
            print(f"  Results returned: {len(complex_results)}")
            print(f"  Filters applied: rating >= 8.0, year >= 1990")
        
        # Test 3: Multi-vector comparison
        print("\n3️⃣ Multi-Vector Comparison")
        
        if sample_result:
            start_time = time.time()
            
            multi_vector_query = """
            SELECT TOP 3
                c.title,
                VectorDistance('cosine', c.content_embedding, @testEmbedding) as content_distance,
                VectorDistance('cosine', c.title_embedding, @testEmbedding) as title_distance,
                VectorDistance('cosine', c.genre_embedding, @testEmbedding) as genre_distance
            FROM c 
            WHERE c.document_type = 'movie' 
                AND IS_DEFINED(c.content_embedding)
                AND IS_DEFINED(c.title_embedding)
                AND IS_DEFINED(c.genre_embedding)
            ORDER BY VectorDistance('cosine', c.content_embedding, @testEmbedding)
            """
            
            multi_results = list(self.container.query_items(
                query=multi_vector_query,
                parameters=[{"name": "@testEmbedding", "value": test_embedding}],
                enable_cross_partition_query=True
            ))
            
            end_time = time.time()
            
            print(f"  Query time: {(end_time - start_time)*1000:.2f} ms")
            print(f"  Multi-vector comparison for {len(multi_results)} movies")
            
            if multi_results:
                for result in multi_results:
                    print(f"    {result['title']}:")
                    print(f"      Content: {result['content_distance']:.4f}")
                    print(f"      Title: {result['title_distance']:.4f}")
                    print(f"      Genre: {result['genre_distance']:.4f}")
    
    def cleanup(self):
        """Clean up test data"""
        print("🗑️ Cleaning up test data...")
        
        try:
            # Delete all documents
            all_docs = self.container.query_items(
                query="SELECT c.id, c.document_type FROM c WHERE c.document_type = 'movie'",
                enable_cross_partition_query=True
            )
            
            for doc in all_docs:
                self.container.delete_item(item=doc['id'], partition_key=doc['document_type'])
            
            print("✅ Test data cleaned up")
            
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")

def main():
    """Main execution function"""
    
    print("🚀 Advanced Vector Operations Demo for Cosmos DB")
    print("=" * 60)
    
    # Initialize operations class
    ops = CosmosVectorOperations()
    
    try:
        # Step 1: Create sample data
        ops.create_sample_documents()
        
        # Step 2: Multi-vector search demonstrations
        print("\n" + "=" * 60)
        print("🔍 Multi-Vector Search Demonstrations")
        print("=" * 60)
        
        # Content-based search
        ops.multi_vector_search("space adventure science fiction", "content", 3)
        
        # Title-based search
        ops.multi_vector_search("matrix inception", "title", 3)
        
        # Genre-based search
        ops.multi_vector_search("crime thriller drama", "genre", 3)
        
        # Step 3: Hybrid search
        print("\n" + "=" * 60)
        print("🔍 Hybrid Vector Search")
        print("=" * 60)
        
        ops.hybrid_vector_search("crime story with complex narrative", {
            "content": 0.6,
            "title": 0.2, 
            "genre": 0.2
        })
        
        # Step 4: Clustering analysis
        print("\n" + "=" * 60)
        print("📊 Vector Clustering Analysis")
        print("=" * 60)
        
        ops.vector_clustering_analysis()
        
        # Step 5: Aggregation queries
        print("\n" + "=" * 60)
        print("📊 Vector Aggregation Operations")
        print("=" * 60)
        
        ops.vector_aggregation_queries()
        
        # Step 6: Performance benchmarks
        print("\n" + "=" * 60)
        print("⚡ Performance Benchmarks")
        print("=" * 60)
        
        ops.vector_performance_benchmarks()
        
        print("\n" + "=" * 60)
        print("✅ Advanced Vector Operations Demo Completed!")
        print("=" * 60)
        
        print("\n🎯 Demonstrated Features:")
        print("  • Multi-vector document storage")
        print("  • Vector search across different embedding types")
        print("  • Hybrid search with weighted combinations")
        print("  • Vector clustering and similarity analysis")
        print("  • Performance benchmarking")
        print("  • Complex vector aggregations")
        
        print(f"\n🔧 Technical Configuration:")
        print(f"  • Database: {DATABASE_NAME}")
        print(f"  • Container: {CONTAINER_NAME}")
        print(f"  • Vector dimensions: 1536 per embedding type")
        print(f"  • Embedding types: content, title, genre")
        print(f"  • Distance functions: cosine, euclidean")
        print(f"  • Index type: quantizedFlat")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
    
    finally:
        # Cleanup (uncomment if needed)
        # ops.cleanup()
        pass

if __name__ == "__main__":
    main()

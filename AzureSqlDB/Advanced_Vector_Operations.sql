-- ========================================
-- Advanced Vector Database Operations
-- Movie Dataset - Azure SQL Database with -- Find recommendations for user based on vector similarity
DECLARE @user_preference_vector VECTOR(1536)ure OpenAI
---- Simulate a RAG query: "Find movies about space adventure"
-- In real RAG, this would use Azure OpenAI embedding model to convert query to vector
DECLARE @rag_query_vector VECTOR(1536)======================================

-- This script demonstrates advanced vector database operations using Azure OpenAI embeddings
-- Run this after executing MovieDB_DDL_Scripts.sql and loading data with Azure OpenAI embeddings

-- ========================================
-- 1. VECTOR DISTANCE METRICS COMPARISON
-- ========================================

PRINT 'Demonstrating Vector Distance Metrics...';

-- Compare different distance metrics for the same movie pair
DECLARE @movie1_vector VECTOR(1536)
DECLARE @movie2_vector VECTOR(1536)

-- Get vectors for two different movies
SELECT TOP 1 @movie1_vector = embedding FROM movie_vectors ORDER BY movie_id;
SELECT TOP 1 @movie2_vector = embedding FROM movie_vectors ORDER BY movie_id DESC;

SELECT 
    'Distance Metrics Comparison' AS analysis_type,
    VECTOR_DISTANCE('cosine', @movie1_vector, @movie2_vector) AS cosine_distance,
    1 - VECTOR_DISTANCE('cosine', @movie1_vector, @movie2_vector) AS cosine_similarity,
    VECTOR_DISTANCE('euclidean', @movie1_vector, @movie2_vector) AS euclidean_distance;

-- ========================================
-- 2. SEMANTIC MOVIE SEARCH SIMULATION
-- ========================================

PRINT 'Semantic Movie Search Examples...';

-- Find action movies (simulate by finding movies similar to known action movie vector)
DECLARE @action_query_vector VECTOR(1536)
SELECT TOP 1 @action_query_vector = embedding 
FROM movie_vectors mv
JOIN movies m ON mv.movie_id = m.movie_id
WHERE m.genres LIKE '%Action%';

IF @action_query_vector IS NOT NULL
BEGIN
    PRINT 'Finding action-like movies:';
    SELECT TOP 5
        mv.title,
        m.genres,
        m.vote_average,
        1 - VECTOR_DISTANCE('cosine', mv.embedding, @action_query_vector) AS similarity_score
    FROM movie_vectors mv
    JOIN movies m ON mv.movie_id = m.movie_id
    ORDER BY similarity_score DESC;
END

-- ========================================
-- 3. VECTOR AGGREGATION OPERATIONS
-- ========================================

PRINT 'Vector Aggregation Examples...';

-- Calculate genre-based vector centroids (average vectors per genre)
-- This simulates creating genre embeddings
WITH GenreMovies AS (
    SELECT 
        CASE 
            WHEN m.genres LIKE '%Action%' THEN 'Action'
            WHEN m.genres LIKE '%Comedy%' THEN 'Comedy'
            WHEN m.genres LIKE '%Drama%' THEN 'Drama'
            WHEN m.genres LIKE '%Family%' THEN 'Family'
            ELSE 'Other'
        END AS genre_category,
        mv.embedding,
        m.title
    FROM movies m
    JOIN movie_vectors mv ON m.movie_id = mv.movie_id
    WHERE m.genres IS NOT NULL
)
SELECT 
    genre_category,
    COUNT(*) AS movie_count
FROM GenreMovies
GROUP BY genre_category
ORDER BY movie_count DESC;

-- ========================================
-- 4. VECTOR SIMILARITY MATRIX
-- ========================================

PRINT 'Creating Vector Similarity Matrix...';

-- Create a similarity matrix for top movies
SELECT 
    m1.title AS movie1,
    m2.title AS movie2,
    ROUND(1 - VECTOR_DISTANCE('cosine', mv1.embedding, mv2.embedding), 4) AS similarity
FROM movie_vectors mv1
JOIN movies m1 ON mv1.movie_id = m1.movie_id
JOIN movie_vectors mv2 ON mv1.movie_id < mv2.movie_id  -- Avoid duplicates
JOIN movies m2 ON mv2.movie_id = m2.movie_id
WHERE m1.popularity > 50 AND m2.popularity > 50  -- Only high popularity movies
ORDER BY similarity DESC;

-- ========================================
-- 5. RECOMMENDATION ENGINE QUERIES
-- ========================================

PRINT 'Recommendation Engine Queries...';

-- Create a recommendation function based on user preferences
-- Simulate user viewing history with movie ratings
CREATE TABLE IF NOT EXISTS user_ratings (
    user_id INT,
    movie_id INT,
    rating FLOAT,
    PRIMARY KEY (user_id, movie_id)
);

-- Insert sample user data
INSERT INTO user_ratings (user_id, movie_id, rating) 
SELECT 1, movie_id, 
       CASE 
           WHEN genres LIKE '%Action%' THEN 4.5
           WHEN genres LIKE '%Comedy%' THEN 3.0
           ELSE 2.5
       END
FROM movies 
WHERE movie_id <= 3;  -- Sample data

-- Find recommendations for user based on vector similarity
DECLARE @user_preference_vector VECTOR(100)

-- Calculate user preference vector (average of liked movies)
WITH UserLikedMovies AS (
    SELECT mv.embedding
    FROM user_ratings ur
    JOIN movie_vectors mv ON ur.movie_id = mv.movie_id
    WHERE ur.user_id = 1 AND ur.rating >= 4.0
)
-- Note: In real scenario, you'd calculate average of vectors
-- For demo, we'll use one of the highly rated movies
SELECT TOP 1 @user_preference_vector = embedding 
FROM movie_vectors mv
JOIN user_ratings ur ON mv.movie_id = ur.movie_id
WHERE ur.user_id = 1 AND ur.rating >= 4.0;

-- Get recommendations
SELECT TOP 5
    mv.title,
    m.overview,
    m.vote_average,
    1 - VECTOR_DISTANCE('cosine', mv.embedding, @user_preference_vector) AS recommendation_score
FROM movie_vectors mv
JOIN movies m ON mv.movie_id = m.movie_id
LEFT JOIN user_ratings ur ON mv.movie_id = ur.movie_id AND ur.user_id = 1
WHERE ur.movie_id IS NULL  -- Movies user hasn't rated
ORDER BY recommendation_score DESC;

-- ========================================
-- 6. VECTOR CLUSTERING SIMULATION
-- ========================================

PRINT 'Vector Clustering Analysis...';

-- Identify potential movie clusters based on vector similarity
-- Find movies that are similar to each other (potential clusters)
WITH SimilarityPairs AS (
    SELECT 
        mv1.movie_id AS movie1_id,
        mv1.title AS movie1_title,
        mv2.movie_id AS movie2_id,
        mv2.title AS movie2_title,
        1 - VECTOR_DISTANCE('cosine', mv1.embedding, mv2.embedding) AS similarity
    FROM movie_vectors mv1
    JOIN movie_vectors mv2 ON mv1.movie_id < mv2.movie_id
    WHERE 1 - VECTOR_DISTANCE('cosine', mv1.embedding, mv2.embedding) > 0.8  -- High similarity threshold
)
SELECT 
    movie1_title,
    movie2_title,
    ROUND(similarity, 4) AS similarity_score
FROM SimilarityPairs
ORDER BY similarity_score DESC;

-- ========================================
-- 7. PERFORMANCE OPTIMIZATION QUERIES
-- ========================================

PRINT 'Performance Analysis...';

-- Check index usage for vector queries
SET STATISTICS IO ON;

-- Test vector similarity query performance
DECLARE @test_vector VECTOR(1536)
SELECT TOP 1 @test_vector = embedding FROM movie_vectors;

-- Measure query performance
DECLARE @start_time DATETIME2 = GETDATE();

SELECT TOP 10
    title,
    1 - VECTOR_DISTANCE('cosine', embedding, @test_vector) AS similarity
FROM movie_vectors
ORDER BY similarity DESC;

DECLARE @end_time DATETIME2 = GETDATE();
SELECT DATEDIFF(MILLISECOND, @start_time, @end_time) AS query_time_ms;

SET STATISTICS IO OFF;

-- ========================================
-- 8. VECTOR QUALITY METRICS
-- ========================================

PRINT 'Vector Quality Analysis...';

-- Analyze vector distribution and quality
SELECT 
    'Vector Statistics' AS metric_type,
    COUNT(*) AS total_vectors,
    AVG(CAST(LEN(CAST(embedding AS NVARCHAR(MAX))) AS FLOAT)) AS avg_vector_size_chars
FROM movie_vectors;

-- Check for potential vector anomalies
WITH VectorMagnitudes AS (
    SELECT 
        movie_id,
        title,
        -- Approximate vector magnitude calculation
        CAST(LEN(CAST(embedding AS NVARCHAR(MAX))) AS FLOAT) AS vector_size
    FROM movie_vectors
)
SELECT 
    'Vector Size Distribution' AS analysis,
    MIN(vector_size) AS min_size,
    MAX(vector_size) AS max_size,
    AVG(vector_size) AS avg_size,
    STDEV(vector_size) AS size_std_dev
FROM VectorMagnitudes;

-- ========================================
-- 9. RAG (RETRIEVAL AUGMENTED GENERATION) SIMULATION
-- ========================================

PRINT 'RAG Pattern Demonstration...';

-- Simulate a RAG query: "Find movies about space adventure"
-- In real RAG, this would use an embedding model to convert query to vector
DECLARE @rag_query_vector VECTOR(100)

-- For demo, use a movie vector that represents "space adventure"
SELECT TOP 1 @rag_query_vector = mv.embedding
FROM movie_vectors mv
JOIN movies m ON mv.movie_id = m.movie_id
WHERE m.overview LIKE '%space%' OR m.title LIKE '%space%'
   OR m.overview LIKE '%adventure%';

-- If no space movies found, use any adventure movie
IF @rag_query_vector IS NULL
BEGIN
    SELECT TOP 1 @rag_query_vector = mv.embedding
    FROM movie_vectors mv
    JOIN movies m ON mv.movie_id = m.movie_id
    WHERE m.genres LIKE '%Adventure%';
END

-- Retrieve relevant context for RAG
SELECT TOP 3
    'RAG Context Retrieval' AS step,
    mv.title,
    m.overview,
    m.genres,
    1 - VECTOR_DISTANCE('cosine', mv.embedding, @rag_query_vector) AS relevance_score
FROM movie_vectors mv
JOIN movies m ON mv.movie_id = m.movie_id
ORDER BY relevance_score DESC;

-- ========================================
-- 9.1. AZURE OPENAI RAG PATTERN SIMULATION
-- ========================================

PRINT 'Azure OpenAI RAG Pattern with GPT-4o Mini...';

-- Simulate a complete RAG workflow for movie recommendations
-- Step 1: User query -> Azure OpenAI Embedding
-- Step 2: Vector search in SQL Database  
-- Step 3: Retrieved context -> Azure OpenAI GPT-4o Mini for generation

-- Example RAG scenario: "Recommend me a sci-fi movie similar to Blade Runner"
DECLARE @user_query NVARCHAR(MAX) = 'Recommend me a sci-fi movie similar to Blade Runner with themes of dystopian future and artificial intelligence'

-- In production, this vector would come from Azure OpenAI text-embedding-3-small
-- For demo, we'll simulate finding a relevant movie vector
DECLARE @query_embedding VECTOR(1536)
SELECT TOP 1 @query_embedding = mv.embedding
FROM movie_vectors mv
JOIN movies m ON mv.movie_id = m.movie_id
WHERE m.overview LIKE '%future%' OR m.overview LIKE '%artificial%' 
   OR m.overview LIKE '%sci%' OR m.genres LIKE '%Science Fiction%'

-- Step 2: Retrieve relevant movies using vector similarity
WITH RelevantMovies AS (
    SELECT TOP 5
        m.title,
        m.overview,
        m.genres,
        m.vote_average,
        m.release_date,
        1 - VECTOR_DISTANCE('cosine', mv.embedding, @query_embedding) AS relevance_score
    FROM movie_vectors mv
    JOIN movies m ON mv.movie_id = m.movie_id
    ORDER BY relevance_score DESC
)
-- Step 3: Format context for GPT-4o Mini
SELECT 
    'RAG Context for GPT-4o Mini' AS context_type,
    STRING_AGG(
        'Title: ' + title + 
        ' | Overview: ' + ISNULL(overview, 'No overview available') + 
        ' | Genres: ' + ISNULL(genres, 'Unknown') + 
        ' | Rating: ' + CAST(vote_average AS NVARCHAR(10)) +
        ' | Year: ' + CAST(YEAR(release_date) AS NVARCHAR(10)), 
        CHAR(13) + CHAR(10) + '---' + CHAR(13) + CHAR(10)
    ) AS formatted_context
FROM RelevantMovies;

-- Simulate the final RAG response structure
SELECT 
    @user_query AS user_query,
    'Based on the retrieved movie data, here are sci-fi recommendations similar to Blade Runner...' AS gpt4o_mini_response_preview,
    'text-embedding-3-small' AS embedding_model_used,
    'gpt-4o-mini' AS generation_model_used,
    GETDATE() AS timestamp;

-- ========================================
-- 10. CLEANUP UTILITY QUERIES
-- ========================================

-- Utility to check vector data integrity
SELECT 
    'Data Integrity Check' AS check_type,
    (SELECT COUNT(*) FROM movies) AS total_movies,
    (SELECT COUNT(*) FROM movie_vectors) AS total_vectors,
    (SELECT COUNT(*) FROM movies m WHERE NOT EXISTS (
        SELECT 1 FROM movie_vectors mv WHERE mv.movie_id = m.movie_id
    )) AS movies_without_vectors;

-- Performance monitoring query
SELECT 
    'Performance Metrics' AS metric_type,
    'Total movies with vectors' AS metric,
    COUNT(*) AS value
FROM movies m
JOIN movie_vectors mv ON m.movie_id = mv.movie_id;

PRINT '========================================';
PRINT 'Advanced vector operations with Azure OpenAI completed!';
PRINT 'Use these patterns for:';
PRINT '- Semantic search with Azure OpenAI embeddings';
PRINT '- Recommendation engines using text-embedding-3-small';
PRINT '- RAG applications with GPT-4o Mini';
PRINT '- Vector quality analysis for 1536-dim embeddings';
PRINT '- Performance optimization for Azure OpenAI vectors';
PRINT '========================================';

-- Additional Azure OpenAI specific recommendations
PRINT 'Azure OpenAI Integration Tips:';
PRINT '- Use text-embedding-3-small (1536 dimensions) for embeddings';
PRINT '- Implement proper retry logic for API calls';
PRINT '- Cache embeddings to reduce API costs';
PRINT '- Use GPT-4o Mini for cost-effective generation';
PRINT '- Consider batch processing for large datasets';
PRINT '========================================';

-- Clean up temporary tables
DROP TABLE IF EXISTS user_ratings;

-- ========================================
-- Azure SQL Database Vector DDL Scripts
-- Movie Dataset Vector Database Demo
-- ========================================

-- Check SQL Server version
SELECT @@version;

-- Enable vector support (if needed)
-- Note: Vector support needs to be enabled at the Azure SQL Database level

-- ========================================
-- 1. DROP EXISTING TABLES (IF NEEDED)
-- ========================================

-- Drop existing tables in correct order (due to foreign key constraints)
IF OBJECT_ID('movie_vectors', 'U') IS NOT NULL
    DROP TABLE movie_vectors;

IF OBJECT_ID('movies', 'U') IS NOT NULL
    DROP TABLE movies;

PRINT 'Existing tables dropped successfully';

-- ========================================
-- 2. CREATE MOVIES MAIN TABLE
-- ========================================

CREATE TABLE movies (
    movie_id INT PRIMARY KEY,
    title NVARCHAR(500) NOT NULL,
    overview NVARCHAR(MAX),
    genres NVARCHAR(MAX),
    release_date DATE,
    budget BIGINT,
    revenue BIGINT,
    runtime FLOAT,
    vote_average FLOAT,
    vote_count INT,
    popularity FLOAT,
    original_language NVARCHAR(10),
    combined_text NVARCHAR(MAX),
    created_at DATETIME2 DEFAULT GETDATE(),
    updated_at DATETIME2 DEFAULT GETDATE()
);

PRINT 'Movies table created successfully';

-- ========================================
-- 3. CREATE MOVIE VECTORS TABLE
-- ========================================

CREATE TABLE movie_vectors (
    movie_id INT PRIMARY KEY,
    title NVARCHAR(500) NOT NULL,
    embedding VECTOR(1536),  -- 1536-dimensional vector for Azure OpenAI text-embedding-3-small
    embedding_model NVARCHAR(100) DEFAULT 'text-embedding-3-small',
    created_at DATETIME2 DEFAULT GETDATE(),
    
    -- Foreign key constraint
    CONSTRAINT FK_movie_vectors_movies 
        FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
        ON DELETE CASCADE
);

PRINT 'Movie vectors table created successfully';

-- ========================================
-- 4. CREATE INDEXES FOR PERFORMANCE
-- ========================================

-- Create index for vector similarity search
-- Note: Vector indexes might have specific syntax in Azure SQL Database
CREATE INDEX idx_movie_vectors_embedding 
ON movie_vectors(embedding);

-- Create indexes on frequently queried columns
CREATE INDEX idx_movies_title ON movies(title);
CREATE INDEX idx_movies_popularity ON movies(popularity DESC);
CREATE INDEX idx_movies_vote_average ON movies(vote_average DESC);
CREATE INDEX idx_movies_release_date ON movies(release_date);

PRINT 'Indexes created successfully';

-- ========================================
-- 5. CREATE SAMPLE DATA FOR TESTING
-- ========================================

-- Insert sample movie data
INSERT INTO movies (movie_id, title, overview, genres, release_date, budget, revenue, runtime, vote_average, vote_count, popularity, original_language, combined_text)
VALUES 
(1, 'Test Movie 1', 'A thrilling action adventure movie', '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]', '2020-01-01', 100000000, 250000000, 120.0, 7.5, 1500, 85.5, 'en', 'Test Movie 1 A thrilling action adventure movie Action Adventure'),
(2, 'Test Movie 2', 'A romantic comedy for the whole family', '[{"id": 35, "name": "Comedy"}, {"id": 10749, "name": "Romance"}]', '2019-06-15', 50000000, 150000000, 95.0, 6.8, 800, 42.3, 'en', 'Test Movie 2 A romantic comedy for the whole family Comedy Romance'),
(3, 'Test Movie 3', 'An animated family film with adventure', '[{"id": 16, "name": "Animation"}, {"id": 10751, "name": "Family"}]', '2021-03-20', 75000000, 200000000, 105.0, 8.2, 2200, 91.7, 'en', 'Test Movie 3 An animated family film with adventure Animation Family');

-- Insert sample vector data (sample Azure OpenAI embeddings format)
INSERT INTO movie_vectors (movie_id, title, embedding)
VALUES 
(1, 'Test Movie 1', CAST('[' + REPLICATE('0.001,', 1535) + '0.001]' AS VECTOR(1536))),
(2, 'Test Movie 2', CAST('[' + REPLICATE('0.002,', 1535) + '0.002]' AS VECTOR(1536))),
(3, 'Test Movie 3', CAST('[' + REPLICATE('0.003,', 1535) + '0.003]' AS VECTOR(1536)));

PRINT 'Sample data inserted successfully';

-- ========================================
-- 6. VERIFY TABLE CREATION
-- ========================================

-- Check table structures
SELECT 
    TABLE_NAME, 
    COLUMN_NAME, 
    DATA_TYPE, 
    IS_NULLABLE,
    COLUMN_DEFAULT
FROM INFORMATION_SCHEMA.COLUMNS 
WHERE TABLE_NAME IN ('movies', 'movie_vectors')
ORDER BY TABLE_NAME, ORDINAL_POSITION;

-- Check row counts
SELECT 'movies' as table_name, COUNT(*) as row_count FROM movies
UNION ALL
SELECT 'movie_vectors' as table_name, COUNT(*) as row_count FROM movie_vectors;

-- ========================================
-- 7. VECTOR SIMILARITY TEST QUERIES
-- ========================================

PRINT 'Testing vector similarity functions...';

-- Test basic vector similarity
DECLARE @test_vector1 VECTOR(1536)
DECLARE @test_vector2 VECTOR(1536)

SELECT @test_vector1 = embedding FROM movie_vectors WHERE movie_id = 1;
SELECT @test_vector2 = embedding FROM movie_vectors WHERE movie_id = 2;

SELECT 
    'Test Vector Similarity' AS test_name,
    1 - VECTOR_DISTANCE('cosine', @test_vector1, @test_vector2) AS cosine_similarity,
    VECTOR_DISTANCE('euclidean', @test_vector1, @test_vector2) AS euclidean_distance;

-- Test vector similarity search
DECLARE @query_vector VECTOR(1536)
SELECT @query_vector = embedding FROM movie_vectors WHERE movie_id = 1;

SELECT 
    mv.title,
    1 - VECTOR_DISTANCE('cosine', mv.embedding, @query_vector) AS similarity_score
FROM movie_vectors mv
WHERE mv.movie_id != 1  -- Exclude the query movie itself
ORDER BY similarity_score DESC;

PRINT 'Vector similarity tests completed successfully';

-- ========================================
-- 8. PERFORMANCE TESTING SETUP
-- ========================================

-- Create a view for easier querying
CREATE VIEW v_movie_similarity AS
SELECT 
    m.movie_id,
    m.title,
    m.overview,
    m.genres,
    m.vote_average,
    m.popularity,
    mv.embedding
FROM movies m
JOIN movie_vectors mv ON m.movie_id = mv.movie_id;

PRINT 'Movie similarity view created';

-- Create a stored procedure for similarity search
CREATE PROCEDURE sp_FindSimilarMovies
    @target_movie_id INT,
    @top_n INT = 5
AS
BEGIN
    DECLARE @target_vector VECTOR(1536)
    
    SELECT @target_vector = embedding 
    FROM movie_vectors 
    WHERE movie_id = @target_movie_id;
    
    IF @target_vector IS NULL
    BEGIN
        PRINT 'Movie not found or no vector available'
        RETURN
    END
    
    SELECT TOP (@top_n)
        mv.movie_id,
        mv.title,
        m.overview,
        m.vote_average,
        m.popularity,
        1 - VECTOR_DISTANCE('cosine', mv.embedding, @target_vector) AS similarity_score
    FROM movie_vectors mv
    JOIN movies m ON mv.movie_id = m.movie_id
    WHERE mv.movie_id != @target_movie_id
    ORDER BY similarity_score DESC;
END;

PRINT 'Stored procedure for similarity search created';

-- Test the stored procedure
PRINT 'Testing similarity search stored procedure...';
EXEC sp_FindSimilarMovies @target_movie_id = 1, @top_n = 2;

PRINT '========================================';
PRINT 'DDL Script execution completed successfully!';
PRINT 'Tables created: movies, movie_vectors';
PRINT 'Indexes created for performance optimization';
PRINT 'Sample data inserted for testing';
PRINT 'Vector similarity functions tested';
PRINT 'Ready for movie dataset loading!';
PRINT '========================================';

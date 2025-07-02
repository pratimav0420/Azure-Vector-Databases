#!/usr/bin/env python3
"""
Azure Vector Databases Demo - Setup Script
This script helps you set up the environment and validate your configuration.
"""

import os
import sys
import subprocess
import importlib.util

def check_python_version():
    """Check if Python version meets requirements."""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required. Current version:", sys.version)
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def check_requirements():
    """Check if required packages are installed."""
    print("\n📦 Checking required packages...")
    
    required_packages = [
        'pandas', 'numpy', 'pyodbc', 'sqlalchemy', 
        'openai', 'tiktoken', 'jupyter'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing_packages.append(package)
            print(f"{package} not found")
        else:
            print(f"{package} installed")
    
    if missing_packages:
        print(f"\n Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def check_config():
    """Check if configuration file exists."""
    print("\nChecking configuration...")
    
    if os.path.exists('config.py'):
        print("config.py found")
        try:
            import config
            # Check if required settings exist
            required_settings = [
                'AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY',
                'SERVER', 'DATABASE', 'USERNAME', 'PASSWORD'
            ]
            
            missing_settings = []
            for setting in required_settings:
                if not hasattr(config, setting) or getattr(config, setting) in ['your-', 'your_']:
                    missing_settings.append(setting)
            
            if missing_settings:
                print(f" Please update these settings in config.py: {', '.join(missing_settings)}")
                return False
            else:
                print("Configuration appears complete")
                return True
                
        except ImportError as e:
            print(f"Error importing config.py: {e}")
            return False
    else:
        print(" config.py not found")
        print("   1. Copy config_template.py to config.py")
        print("   2. Update with your Azure credentials")
        return False

def check_dataset():
    """Check if dataset files exist."""
    print("\nChecking dataset...")
    
    dataset_path = os.path.join('data', 'moviesdataset')
    required_files = ['movies_metadata.csv', 'ratings_small.csv']
    
    if not os.path.exists(dataset_path):
        print(f" Dataset directory not found: {dataset_path}")
        print("   Create the directory and download the movies dataset")
        return False
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(dataset_path, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
        else:
            print(f"✅ {file} found")
    
    if missing_files:
        print(f" Missing dataset files: {', '.join(missing_files)}")
        print("   Download from: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset")
        return False
    
    return True

def test_azure_connection():
    """Test Azure OpenAI connection."""
    print("\nTesting Azure OpenAI connection...")
    
    try:
        import config
        from openai import AzureOpenAI
        
        client = AzureOpenAI(
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=getattr(config, 'AZURE_OPENAI_API_VERSION', '2024-06-01')
        )
        
        # Test embedding
        response = client.embeddings.create(
            input="test connection",
            model=getattr(config, 'EMBEDDING_MODEL', 'text-embedding-3-small')
        )
        
        print("✅ Azure OpenAI connection successful")
        print(f"   Embedding dimensions: {len(response.data[0].embedding)}")
        return True
        
    except Exception as e:
        print(f" Azure OpenAI connection failed: {e}")
        print("   Check your endpoint, API key, and model deployments")
        return False

def test_database_connection():
    """Test Azure SQL Database connection."""
    print("\n🗄️  Testing Azure SQL Database connection...")
    
    try:
        import config
        import pyodbc
        
        connection_string = f"""
            Driver={{ODBC Driver 18 for SQL Server}};
            Server={config.SERVER};
            Database={config.DATABASE};
            Uid={config.USERNAME};
            Pwd={config.PASSWORD};
            Encrypt=yes;
            TrustServerCertificate=no;
            Connection Timeout=30;
        """
        
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        cursor.execute("SELECT @@VERSION")
        version = cursor.fetchone()[0]
        
        print("✅ Azure SQL Database connection successful")
        print(f"   Database version: {version.split('Microsoft SQL Server')[1].split('-')[0].strip()}")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("   Check your server, database, username, and password")
        return False

def create_sample_config():
    """Create a sample config file if it doesn't exist."""
    if not os.path.exists('config.py') and os.path.exists('config_template.py'):
        print("\n📝 Creating config.py from template...")
        import shutil
        shutil.copy('config_template.py', 'config.py')
        print("✅ config.py created")
        print("   Please update it with your Azure credentials")

def main():
    """Run all setup checks."""
    print("🚀 Azure Vector Databases Demo - Setup Validation")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_requirements),
        ("Configuration", check_config),
        ("Dataset Files", check_dataset),
    ]
    
    # Run basic checks first
    all_basic_passed = True
    for name, check_func in checks:
        if not check_func():
            all_basic_passed = False
    
    # Only test connections if basic checks pass
    if all_basic_passed:
        connection_checks = [
            ("Azure OpenAI Connection", test_azure_connection),
            ("Database Connection", test_database_connection),
        ]
        
        for name, check_func in connection_checks:
            check_func()
    
    print("\n" + "=" * 60)
    
    if all_basic_passed:
        print("🎉 Setup validation complete!")
        print("\nNext steps:")
        print("1. Open MovieDataset_VectorDB_Demo.ipynb in Jupyter")
        print("2. Run the cells step by step")
        print("3. Explore the vector database capabilities")
    else:
        print("⚠️  Please resolve the issues above before proceeding")
        create_sample_config()

if __name__ == "__main__":
    main()

"""
MongoDB vCore Demo 2.1 - Simple Completion
Equivalent to SQL Demo 2.1 - Simple Completion.sql

This script demonstrates Azure OpenAI chat completion with conversation history stored in MongoDB vCore
"""

from pymongo import MongoClient
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import time
import json

# Load environment variables
load_dotenv()

# Configuration
MONGODB_CONNECTION_STRING = os.getenv("MONGODB_CONNECTION_STRING")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o")

DATABASE_NAME = "ChatCompletionDB"
COLLECTION_NAME = "conversations"

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
print(f"Using generation model: {GENERATION_MODEL}")

# Clear existing conversations for fresh demo
collection.delete_many({})
print("Cleared existing conversation data")

# Function to get chat completion from Azure OpenAI
def get_chat_completion(messages, model=GENERATION_MODEL, max_tokens=500):
    """Get chat completion from Azure OpenAI"""
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting chat completion: {e}")
        return None

# Function to store conversation in MongoDB
def store_conversation(conversation_id, user_message, assistant_response, messages_context=None):
    """Store conversation turn in MongoDB"""
    
    conversation_doc = {
        "conversation_id": conversation_id,
        "timestamp": time.time(),
        "user_message": user_message,
        "assistant_response": assistant_response,
        "model_used": GENERATION_MODEL,
        "messages_context": messages_context or []
    }
    
    try:
        result = collection.insert_one(conversation_doc)
        print(f"✅ Stored conversation turn with ID: {result.inserted_id}")
        return result.inserted_id
    except Exception as e:
        print(f"❌ Error storing conversation: {e}")
        return None

# Function to retrieve conversation history
def get_conversation_history(conversation_id, limit=10):
    """Retrieve conversation history from MongoDB"""
    
    try:
        cursor = collection.find(
            {"conversation_id": conversation_id}
        ).sort("timestamp", 1).limit(limit)
        
        history = []
        for doc in cursor:
            history.append({
                "timestamp": doc["timestamp"],
                "user_message": doc["user_message"],
                "assistant_response": doc["assistant_response"]
            })
        
        return history
    except Exception as e:
        print(f"❌ Error retrieving conversation history: {e}")
        return []

# Function to build messages array from conversation history
def build_messages_from_history(conversation_id, new_user_message, system_prompt=None):
    """Build messages array for OpenAI API from conversation history"""
    
    messages = []
    
    # Add system prompt if provided
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Get conversation history
    history = get_conversation_history(conversation_id, limit=5)  # Last 5 exchanges
    
    # Add historical messages
    for turn in history:
        messages.append({"role": "user", "content": turn["user_message"]})
        messages.append({"role": "assistant", "content": turn["assistant_response"]})
    
    # Add new user message
    messages.append({"role": "user", "content": new_user_message})
    
    return messages

# Demo: Simple chat completion without history
print("\n🤖 Demo 1: Simple Chat Completion (No History)")
simple_message = "Hello! Tell me a fun fact about movies."

simple_messages = [
    {"role": "system", "content": "You are a helpful assistant that knows about movies."},
    {"role": "user", "content": simple_message}
]

response1 = get_chat_completion(simple_messages)
if response1:
    print(f"User: {simple_message}")
    print(f"Assistant: {response1}")
    
    # Store this conversation
    store_conversation("demo_simple", simple_message, response1, simple_messages)
else:
    print("❌ Failed to get simple completion")

# Demo: Multi-turn conversation with history
print("\n💬 Demo 2: Multi-turn Conversation with History")

conversation_id = "demo_conversation_1"
system_prompt = "You are a knowledgeable movie expert assistant. Provide helpful and engaging responses about movies, actors, and film industry topics."

# First turn
user_msg1 = "What are some great science fiction movies from the 1980s?"
messages1 = build_messages_from_history(conversation_id, user_msg1, system_prompt)

print(f"\n🎬 Turn 1:")
print(f"User: {user_msg1}")

response1 = get_chat_completion(messages1)
if response1:
    print(f"Assistant: {response1}")
    store_conversation(conversation_id, user_msg1, response1, messages1)
else:
    print("❌ Failed to get response for turn 1")

# Short delay between turns
time.sleep(1)

# Second turn - follow up question
user_msg2 = "Which of those movies had the best special effects for its time?"
messages2 = build_messages_from_history(conversation_id, user_msg2, system_prompt)

print(f"\n🎬 Turn 2:")
print(f"User: {user_msg2}")

response2 = get_chat_completion(messages2)
if response2:
    print(f"Assistant: {response2}")
    store_conversation(conversation_id, user_msg2, response2, messages2)
else:
    print("❌ Failed to get response for turn 2")

# Short delay between turns
time.sleep(1)

# Third turn - another follow up
user_msg3 = "Tell me more about the director of that movie."
messages3 = build_messages_from_history(conversation_id, user_msg3, system_prompt)

print(f"\n🎬 Turn 3:")
print(f"User: {user_msg3}")

response3 = get_chat_completion(messages3)
if response3:
    print(f"Assistant: {response3}")
    store_conversation(conversation_id, user_msg3, response3, messages3)
else:
    print("❌ Failed to get response for turn 3")

# Demo: Conversation context analysis
print("\n📊 Demo 3: Conversation Analysis")

# Retrieve and display conversation history
print("📝 Full Conversation History:")
full_history = get_conversation_history(conversation_id)

for i, turn in enumerate(full_history, 1):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(turn['timestamp']))
    print(f"\nTurn {i} ({timestamp}):")
    print(f"User: {turn['user_message']}")
    print(f"Assistant: {turn['assistant_response'][:100]}...")  # Truncate for display

# Demo: Topic-specific conversations
print("\n🎭 Demo 4: Topic-Specific Conversations")

# Horror movies conversation
horror_conversation_id = "horror_movies_chat"
horror_system_prompt = "You are a horror movie expert. Provide recommendations and insights about horror films."

horror_queries = [
    "What are some classic horror movies everyone should watch?",
    "Which modern horror films have innovative storytelling?",
    "What makes a horror movie truly scary?"
]

for i, query in enumerate(horror_queries, 1):
    print(f"\n🧛 Horror Chat Turn {i}:")
    print(f"User: {query}")
    
    messages = build_messages_from_history(horror_conversation_id, query, horror_system_prompt)
    response = get_chat_completion(messages)
    
    if response:
        print(f"Assistant: {response[:200]}...")  # Truncate for display
        store_conversation(horror_conversation_id, query, response, messages)
    else:
        print("❌ Failed to get response")
    
    time.sleep(1)  # Rate limiting

# Database statistics
print("\n📊 MongoDB Collection Statistics:")
total_conversations = collection.count_documents({})
unique_conversation_ids = len(collection.distinct("conversation_id"))

print(f"Total conversation turns stored: {total_conversations}")
print(f"Unique conversations: {unique_conversation_ids}")

# Show conversation breakdown by ID
conversation_counts = collection.aggregate([
    {"$group": {"_id": "$conversation_id", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}}
])

print("\nConversation breakdown:")
for conv in conversation_counts:
    print(f"  {conv['_id']}: {conv['count']} turns")

# Show models used
models_used = collection.distinct("model_used")
print(f"\nModels used: {models_used}")

# Demo: Search conversations by content
print("\n🔍 Demo 5: Search Conversations by Content")

# Search for conversations mentioning specific topics
search_term = "special effects"
print(f"Searching for conversations mentioning '{search_term}':")

search_results = collection.find({
    "$or": [
        {"user_message": {"$regex": search_term, "$options": "i"}},
        {"assistant_response": {"$regex": search_term, "$options": "i"}}
    ]
}, {"conversation_id": 1, "user_message": 1, "assistant_response": 1}).limit(3)

for result in search_results:
    print(f"\nConversation ID: {result['conversation_id']}")
    print(f"User: {result['user_message']}")
    print(f"Assistant: {result['assistant_response'][:150]}...")

# Function to export conversation to JSON
def export_conversation_to_json(conversation_id, filename=None):
    """Export a conversation to JSON file"""
    
    if not filename:
        filename = f"conversation_{conversation_id}_{int(time.time())}.json"
    
    history = get_conversation_history(conversation_id, limit=100)
    
    export_data = {
        "conversation_id": conversation_id,
        "exported_at": time.time(),
        "total_turns": len(history),
        "conversation": history
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Exported conversation '{conversation_id}' to {filename}")
        return filename
    except Exception as e:
        print(f"❌ Error exporting conversation: {e}")
        return None

# Export sample conversation
print("\n💾 Demo 6: Export Conversation")
exported_file = export_conversation_to_json(conversation_id)

print("\n✅ Chat completion demo completed!")
print("This demo showed:")
print("• Azure OpenAI GPT-4o chat completion integration")
print("• Storing conversation history in MongoDB documents")
print("• Multi-turn conversations with context awareness")
print("• Conversation retrieval and analysis")
print("• Topic-specific conversation management")
print("• Content search within conversations")
print("• Conversation export capabilities")
print("• Conversation statistics and analytics")

# Cleanup (uncomment to clean up test data)
# collection.delete_many({})
# print("🧹 Cleaned up conversation data")

# Close connection
mongo_client.close()

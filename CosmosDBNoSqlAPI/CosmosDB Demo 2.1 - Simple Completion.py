"""
Cosmos DB Demo 2.1 - Simple Completion
Equivalent to SQL Demo 2.1 - Simple Completion.sql

This script demonstrates Azure OpenAI completion integration with Cosmos DB NoSQL API
"""

from azure.cosmos import CosmosClient, PartitionKey
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import time
import json

# Load environment variables
load_dotenv()

# Configuration
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o")

DATABASE_NAME = "CompletionTestDB"
CONTAINER_NAME = "ChatHistory"

# Initialize clients
cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION
)

print("Initialized Azure Cosmos DB and Azure OpenAI clients")
print(f"Using generation model: {GENERATION_MODEL}")

# Create database
database = cosmos_client.create_database_if_not_exists(id=DATABASE_NAME)
print(f"Database '{DATABASE_NAME}' ready")

# Create container for chat history
container = database.create_container_if_not_exists(
    id=CONTAINER_NAME,
    partition_key=PartitionKey(path="/conversation_id"),
    offer_throughput=400
)
print(f"Container '{CONTAINER_NAME}' ready")

def get_completion(messages, max_tokens=1000):
    """Get completion from Azure OpenAI"""
    try:
        response = openai_client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        return {
            "content": response.choices[0].message.content,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    except Exception as e:
        print(f"Error getting completion: {e}")
        return None

def save_chat_message(conversation_id, user_prompt, assistant_response, prompt_tokens, completion_tokens):
    """Save chat message to Cosmos DB"""
    try:
        # Generate unique message ID
        message_id = f"{conversation_id}_{int(time.time() * 1000)}"
        
        doc = {
            "id": message_id,
            "conversation_id": conversation_id,
            "user_prompt": user_prompt,
            "assistant_response": assistant_response,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "document_type": "chat_message"
        }
        
        container.create_item(doc)
        print(f"💾 Saved chat message: {message_id}")
        return message_id
        
    except Exception as e:
        print(f"❌ Error saving chat message: {e}")
        return None

def get_conversation_history(conversation_id, limit=10):
    """Get conversation history from Cosmos DB"""
    try:
        history_query = """
        SELECT TOP @limit
            c.user_prompt,
            c.assistant_response,
            c.timestamp
        FROM c 
        WHERE c.conversation_id = @conversationId 
            AND c.document_type = 'chat_message'
        ORDER BY c.timestamp ASC
        """
        
        results = list(container.query_items(
            query=history_query,
            parameters=[
                {"name": "@conversationId", "value": conversation_id},
                {"name": "@limit", "value": limit}
            ],
            enable_cross_partition_query=True
        ))
        
        return results
        
    except Exception as e:
        print(f"❌ Error getting conversation history: {e}")
        return []

# Clear existing chat history for demo
print("\n=== Clearing Previous Chat History ===")
try:
    all_messages = container.query_items(
        query="SELECT c.id, c.conversation_id FROM c WHERE c.document_type = 'chat_message'",
        enable_cross_partition_query=True
    )
    
    deleted_count = 0
    for msg in all_messages:
        container.delete_item(item=msg['id'], partition_key=msg['conversation_id'])
        deleted_count += 1
    
    print(f"🗑️ Deleted {deleted_count} previous messages")
except Exception as e:
    print(f"Note: {e}")

# Demo 1: Simple completion
print("\n=== Demo 1: Simple Completion ===")

conversation_id = "demo_conversation_1"
system_prompt = "You are an AI assistant that helps people find information."
user_prompt = "What is the biggest mammal?"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

print(f"💭 User: {user_prompt}")

# Get completion
completion_result = get_completion(messages)

if completion_result:
    print(f"🤖 Assistant: {completion_result['content']}")
    print(f"📊 Tokens - Prompt: {completion_result['prompt_tokens']}, Completion: {completion_result['completion_tokens']}")
    
    # Save to chat history
    save_chat_message(
        conversation_id=conversation_id,
        user_prompt=user_prompt,
        assistant_response=completion_result['content'],
        prompt_tokens=completion_result['prompt_tokens'],
        completion_tokens=completion_result['completion_tokens']
    )

# Demo 2: Continuation with context
print("\n=== Demo 2: Continuation with Context ===")

# Get conversation history
history = get_conversation_history(conversation_id)
print(f"📚 Retrieved {len(history)} previous messages")

# Build context from history
context_messages = [{"role": "system", "content": system_prompt}]

for msg in history:
    context_messages.append({"role": "user", "content": msg['user_prompt']})
    context_messages.append({"role": "assistant", "content": msg['assistant_response']})

# New user prompt
user_prompt_2 = "What is the smallest?"
context_messages.append({"role": "user", "content": user_prompt_2})

print(f"💭 User: {user_prompt_2}")
print(f"🔄 Using context from {len(history)} previous messages")

# Get completion with context
completion_result_2 = get_completion(context_messages)

if completion_result_2:
    print(f"🤖 Assistant: {completion_result_2['content']}")
    print(f"📊 Tokens - Prompt: {completion_result_2['prompt_tokens']}, Completion: {completion_result_2['completion_tokens']}")
    
    # Save to chat history
    save_chat_message(
        conversation_id=conversation_id,
        user_prompt=user_prompt_2,
        assistant_response=completion_result_2['content'],
        prompt_tokens=completion_result_2['prompt_tokens'],
        completion_tokens=completion_result_2['completion_tokens']
    )

# Demo 3: Multiple conversation threads
print("\n=== Demo 3: Multiple Conversation Threads ===")

conversation_topics = [
    {"id": "science_chat", "prompt": "Explain photosynthesis in simple terms"},
    {"id": "cooking_chat", "prompt": "How do I make perfect pasta?"},
    {"id": "tech_chat", "prompt": "What is machine learning?"}
]

for topic in conversation_topics:
    print(f"\n--- Conversation: {topic['id']} ---")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides clear, concise answers."},
        {"role": "user", "content": topic['prompt']}
    ]
    
    print(f"💭 User: {topic['prompt']}")
    
    result = get_completion(messages, max_tokens=150)  # Shorter responses for demo
    
    if result:
        print(f"🤖 Assistant: {result['content'][:200]}..." if len(result['content']) > 200 else f"🤖 Assistant: {result['content']}")
        
        save_chat_message(
            conversation_id=topic['id'],
            user_prompt=topic['prompt'],
            assistant_response=result['content'],
            prompt_tokens=result['prompt_tokens'],
            completion_tokens=result['completion_tokens']
        )

# Analytics: Chat statistics
print("\n=== Chat Analytics ===")

# Total messages and tokens
stats_query = """
SELECT 
    COUNT(1) as total_messages,
    SUM(c.prompt_tokens) as total_prompt_tokens,
    SUM(c.completion_tokens) as total_completion_tokens,
    SUM(c.total_tokens) as total_tokens,
    AVG(c.prompt_tokens) as avg_prompt_tokens,
    AVG(c.completion_tokens) as avg_completion_tokens
FROM c 
WHERE c.document_type = 'chat_message'
"""

stats = list(container.query_items(query=stats_query, enable_cross_partition_query=True))
if stats and stats[0]['total_messages'] > 0:
    stat = stats[0]
    print(f"📈 Total Messages: {stat['total_messages']}")
    print(f"📈 Total Tokens: {stat['total_tokens']} (Prompt: {stat['total_prompt_tokens']}, Completion: {stat['total_completion_tokens']})")
    print(f"📈 Average Tokens per Message: {stat['avg_prompt_tokens']:.1f} prompt, {stat['avg_completion_tokens']:.1f} completion")

# Conversation breakdown
conv_stats_query = """
SELECT 
    c.conversation_id,
    COUNT(1) as message_count,
    SUM(c.total_tokens) as conversation_tokens
FROM c 
WHERE c.document_type = 'chat_message'
GROUP BY c.conversation_id
ORDER BY conversation_tokens DESC
"""

conv_stats = list(container.query_items(query=conv_stats_query, enable_cross_partition_query=True))
print(f"\n📊 Conversation Breakdown:")
for conv in conv_stats:
    print(f"  {conv['conversation_id']}: {conv['message_count']} messages, {conv['conversation_tokens']} tokens")

# Recent conversations
print("\n=== Recent Conversations ===")
recent_query = """
SELECT TOP 5
    c.conversation_id,
    c.user_prompt,
    c.assistant_response,
    c.timestamp,
    c.total_tokens
FROM c 
WHERE c.document_type = 'chat_message'
ORDER BY c.timestamp DESC
"""

recent_messages = list(container.query_items(query=recent_query, enable_cross_partition_query=True))
for msg in recent_messages:
    print(f"\n🕒 {msg['timestamp']} [{msg['conversation_id']}]")
    print(f"💭 User: {msg['user_prompt'][:100]}...")
    print(f"🤖 Assistant: {msg['assistant_response'][:100]}...")
    print(f"📊 Tokens: {msg['total_tokens']}")

# Demo 4: Structured conversation flow
print("\n=== Demo 4: Structured Conversation Flow ===")

structured_conversation = "structured_demo"
conversation_flow = [
    "I want to learn about renewable energy",
    "Tell me more about solar power",
    "What are the advantages and disadvantages?",
    "How much does it cost to install solar panels?"
]

print("🔄 Simulating a structured conversation flow...")

for i, prompt in enumerate(conversation_flow, 1):
    print(f"\n--- Step {i} ---")
    
    # Get conversation history for context
    history = get_conversation_history(structured_conversation)
    
    # Build messages with context
    messages = [{"role": "system", "content": "You are an expert on renewable energy and sustainability."}]
    
    for msg in history:
        messages.append({"role": "user", "content": msg['user_prompt']})
        messages.append({"role": "assistant", "content": msg['assistant_response']})
    
    messages.append({"role": "user", "content": prompt})
    
    print(f"💭 User: {prompt}")
    print(f"📚 Context: {len(history)} previous messages")
    
    result = get_completion(messages, max_tokens=200)
    
    if result:
        print(f"🤖 Assistant: {result['content']}")
        
        save_chat_message(
            conversation_id=structured_conversation,
            user_prompt=prompt,
            assistant_response=result['content'],
            prompt_tokens=result['prompt_tokens'],
            completion_tokens=result['completion_tokens']
        )

# Final statistics
print("\n=== Final Statistics ===")
final_stats = list(container.query_items(query=stats_query, enable_cross_partition_query=True))
if final_stats and final_stats[0]['total_messages'] > 0:
    stat = final_stats[0]
    print(f"🎯 Demo completed with {stat['total_messages']} total messages")
    print(f"🎯 Total tokens consumed: {stat['total_tokens']}")
    print(f"🎯 Average conversation length: {stat['avg_prompt_tokens'] + stat['avg_completion_tokens']:.1f} tokens")

print(f"\n✅ Completion demo finished!")
print(f"Model used: {GENERATION_MODEL}")
print("Demonstrates: chat completion, conversation history, context management with Cosmos DB")

# Cleanup (uncomment to delete test data)
print("\n=== Cleanup ===")
print("To clean up test data, uncomment the cleanup section below")

# # Delete all chat messages
# print("Deleting chat history...")
# all_messages = container.query_items(
#     query="SELECT c.id, c.conversation_id FROM c WHERE c.document_type = 'chat_message'",
#     enable_cross_partition_query=True
# )
# 
# for msg in all_messages:
#     try:
#         container.delete_item(item=msg['id'], partition_key=msg['conversation_id'])
#     except Exception as e:
#         print(f"Error deleting message: {e}")
# 
# # Delete container
# database.delete_container(container)
# print(f"Container '{CONTAINER_NAME}' deleted")
# 
# # Delete database  
# cosmos_client.delete_database(database)
# print(f"Database '{DATABASE_NAME}' deleted")

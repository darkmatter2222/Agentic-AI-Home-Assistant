import asyncio
import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import ChatMessageContent, AuthorRole
from semantic_kernel.functions import kernel_function


# Load environment variables
load_dotenv()

class MemoryStoragePlugin:
    """Plugin for storing long-term memories to a file."""
    
    def __init__(self, memory_file_path: str = "home_memories.json"):
        self.memory_file_path = Path(memory_file_path)
        self._ensure_memory_file_exists()
    
    def _ensure_memory_file_exists(self):
        """Ensure the memory file exists."""
        if not self.memory_file_path.exists():
            self.memory_file_path.write_text("[]", encoding="utf-8")
    
    @kernel_function
    def store_memory(self, information: str, context: str = "") -> str:
        """Store any information with optional context. No need for specific keys."""
        try:
            # Load existing memories
            memories = json.loads(self.memory_file_path.read_text(encoding="utf-8"))
            
            # Create new memory entry with timestamp and context
            memory_entry = {
                "information": information,
                "context": context,
                "timestamp": datetime.now().isoformat(),
                "id": len(memories) + 1,
                "date_readable": datetime.now().strftime('%B %d, %Y at %I:%M %p')
            }
            
            # Add to memories
            memories.append(memory_entry)
            
            # Save back to file
            self.memory_file_path.write_text(json.dumps(memories, indent=2), encoding="utf-8")
            
            print(f"📝 Memory stored: {information}")
            return f"I've remembered: {information}"
            
        except Exception as e:
            return f"Error storing memory: {e}"


class MemoryRetrievalPlugin:
    """Plugin for retrieving information from long-term memory."""
    
    def __init__(self, memory_file_path: str = "home_memories.json"):
        self.memory_file_path = Path(memory_file_path)
    
    @kernel_function
    def search_memory(self, query: str) -> str:
        """Search for information in long-term memory using natural language."""
        try:
            if not self.memory_file_path.exists():
                return "I don't have any memories stored yet."
            
            memories = json.loads(self.memory_file_path.read_text(encoding="utf-8"))
            
            if not memories:
                return "I don't have any memories stored yet."
            
            # Search for relevant memories with flexible matching
            query_lower = query.lower()
            relevant_memories = []
            
            # Look for keywords in the query and match against information and context
            search_terms = [
                "family", "wife", "husband", "kids", "children", "child", "son", "daughter",
                "pets", "dogs", "cats", "animals", "home", "house", "work", "job", 
                "hobbies", "interests", "preferences", "schedule", "routine"
            ]
            
            for memory in memories:
                # Direct text matching
                if (query_lower in memory["information"].lower() or 
                    query_lower in memory.get("context", "").lower()):
                    relevant_memories.append(memory)
                # Semantic matching for family queries
                elif any(term in query_lower for term in ["family", "life", "about me", "who am i"]):
                    relevant_memories.append(memory)
            
            if not relevant_memories:
                return f"I don't have any specific memories about '{query}' yet. Feel free to tell me more!"
            
            # Format results in a natural way
            if len(relevant_memories) == 1:
                memory = relevant_memories[0]
                return f"I remember that {memory['information']}. (From our conversation on {memory['date_readable']})"
            else:
                # Group and present multiple memories naturally
                info_parts = []
                for memory in relevant_memories[-5:]:  # Show most recent 5
                    info_parts.append(memory['information'])
                
                return f"Here's what I remember about you: {', '.join(info_parts)}."
            
        except Exception as e:
            return f"Error searching memory: {e}"
    
    @kernel_function
    def get_all_memories(self) -> str:
        """Get all stored memories about the user."""
        try:
            if not self.memory_file_path.exists():
                return "I don't have any memories stored yet."
            
            memories = json.loads(self.memory_file_path.read_text(encoding="utf-8"))
            
            if not memories:
                return "I don't have any memories stored yet."
            
            # Present all memories in a natural narrative
            info_parts = []
            for memory in memories:
                info_parts.append(memory['information'])
            
            return f"Here's everything I know about you: {' '.join(info_parts)}"
            
        except Exception as e:
            return f"Error retrieving memories: {e}"


class HomeAssistantPlugin:
    """Plugin for home assistant functions."""
    
    @kernel_function
    def get_current_time(self) -> str:
        """Get the current date and time."""
        now = datetime.now()
        return f"Current date and time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"
    
    @kernel_function
    def note_household_info(self, info_type: str, details: str) -> str:
        """Note information about household items, preferences, or routines."""
        print(f"🏠 Noting household info: {info_type} -> {details}")
        return f"Noted household information - {info_type}: {details}"


def create_openai_service():
    """Create OpenAI chat completion service."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4")
    org_id = os.getenv("OPENAI_ORG_ID")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Basic service configuration
    service_kwargs = {
        "api_key": api_key,
        "ai_model_id": model
    }
    
    # Only add org_id if it's provided (not needed for personal accounts)
    if org_id and org_id.strip() and not org_id.startswith("#"):
        service_kwargs["org_id"] = org_id
        print(f"🏢 Using organization ID: {org_id}")
    else:
        print("👤 Using personal OpenAI account (no organization ID)")
    
    print(f"🤖 Using model: {model}")
    
    return OpenAIChatCompletion(**service_kwargs)


def create_niko_agent():
    """Create Niko agent with all capabilities."""
    openai_service = create_openai_service()
    
    # Create Niko with integrated memory capabilities
    niko_agent = ChatCompletionAgent(
        name="Niko",
        description="Your friendly home manager who helps with household tasks, routines, and remembering important information.",
        instructions="""You are Niko, a friendly and helpful home manager assistant. 

Your main abilities:
- Have natural conversations about home and family
- Help with household organization and routines
- Automatically store and retrieve important information using your memory functions

MEMORY APPROACH:
You have two main memory functions:
1. store_memory(information, context) - Store any information naturally
2. search_memory(query) - Search through memories with natural language
3. get_all_memories() - Get everything you know about the user

WHEN TO STORE INFORMATION:
Store information when users mention anything about themselves, their family, or their life:
- Personal details (name, family, pets, etc.)
- Preferences and habits
- Schedules and routines
- Important facts about their life

HOW TO STORE:
Simply call store_memory() with the information as it was told to you.

Examples:
- User says "My name is Ryan" -> Call: store_memory("My name is Ryan", "personal introduction")
- User says "I have three kids" -> Call: store_memory("I have three kids", "family information")
- User says "I have a wife and I'm happy" -> Call: store_memory("I have a wife and I'm happy", "family and personal state")

HOW TO RETRIEVE:
When asked about the user or their life:
- For specific queries, use search_memory() with relevant terms
- For general "tell me about me" questions, use get_all_memories()
- Always present information naturally and conversationally

Be warm, personal, and remember that you're helping manage their home life. Store information naturally as conversations flow.""",
        service=openai_service,
        plugins=[HomeAssistantPlugin(), MemoryStoragePlugin(), MemoryRetrievalPlugin()],
    )
    
    return niko_agent


async def main():
    """Main function to run Niko's Home Assistant."""
    print("🏠 Starting Niko's Home Assistant")
    print("=" * 50)
    print("👋 Meet Niko - Your friendly home manager")
    print("💾 Niko can store and remember important information")
    print("🔍 Niko can search through your stored memories")
    print("=" * 50)
    print("Tell Niko about your household, routines, preferences, and important information.")
    print("Niko will remember everything and help you manage your home!")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.")
    print("=" * 50)
    
    try:
        # Create Niko agent
        niko = create_niko_agent()
        
        # Initial greeting from Niko
        print("\n🤖 Niko: Hello! I'm Niko, your home manager assistant. I'm here to help you")
        print("organize your household, remember important information, and make your home")
        print("life easier. What would you like to tell me about your home or family today?")
        
        # Main interaction loop
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n👋 Niko: Goodbye! I'll remember everything we discussed for next time!")
                    break
                
                if not user_input:
                    continue
                
                # Create user message
                user_message = ChatMessageContent(role=AuthorRole.USER, content=user_input)
                
                # Get response from Niko
                responses = []
                async for response in niko.invoke([user_message]):
                    if hasattr(response, 'content') and response.content:
                        content = str(response.content).strip()
                        if content:
                            responses.append(content)
                
                # Print responses
                if responses:
                    for response in responses:
                        print(f"\n🤖 Niko: {response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Niko: Goodbye! Take care!")
                break
            except Exception as e:
                print(f"\n❌ Error during conversation: {e}")
                print("Please try again or type 'quit' to exit.")
        
    except Exception as e:
        print(f"❌ Error starting Niko's Home Assistant: {e}")
        print("\nPlease check your .env file and ensure your OpenAI credentials are correct.")


if __name__ == "__main__":
    asyncio.run(main())

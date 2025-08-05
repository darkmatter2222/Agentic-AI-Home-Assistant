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
import threading  # for asynchronous file writes


# Load environment variables
load_dotenv()

class MemoryStoragePlugin:
    """Plugin for storing long-term memories to a file."""
    
    def __init__(self, memory_file_path: str = "home_memories.json"):
        self.memory_file_path = Path(memory_file_path)
        self._ensure_memory_file_exists()
        # Load all existing memories into memory for fast access
        self.memories = json.loads(self.memory_file_path.read_text(encoding="utf-8"))
    
    def _ensure_memory_file_exists(self):
        """Ensure the memory file exists."""
        if not self.memory_file_path.exists():
            self.memory_file_path.write_text("[]", encoding="utf-8")
    
    def _save_memories(self):
        """Write the in-memory memories list to disk."""
        try:
            self.memory_file_path.write_text(json.dumps(self.memories, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"⚠️ Error saving memories: {e}")
    
    @kernel_function
    def store_memory(self, information: str, context: str = "") -> str:
        """Store any information with optional context. No need for specific keys."""
        try:
            # Create new memory entry
            memory_entry = {
                "information": information,
                "context": context,
                "timestamp": datetime.now().isoformat(),
                "id": len(self.memories) + 1,
                "date_readable": datetime.now().strftime('%B %d, %Y at %I:%M %p')
            }
            # Add to in-memory list
            self.memories.append(memory_entry)
            # Asynchronously save to disk
            threading.Thread(target=self._save_memories, daemon=True).start()
            
            print(f"📝 Memory stored: {information}")
            return f"I've remembered: {information}"
            
        except Exception as e:
            return f"Error storing memory: {e}"


class MemoryRetrievalPlugin:
    """Plugin for retrieving information from long-term memory."""
    
    def __init__(self, memory_file_path: str = "home_memories.json"):
        self.memory_file_path = Path(memory_file_path)
        # Load all memories into memory for fast access
        self.memories = json.loads(self.memory_file_path.read_text(encoding="utf-8"))
    
    @kernel_function
    def search_memory(self, query: str) -> str:
        """Search for information in long-term memory using natural language."""
        try:
            # Use in-memory list of memories
            if not self.memories:
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
            
            for memory in self.memories:
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
                # Show most recent 5 memories
                for memory in relevant_memories[-5:]:
                    info_parts.append(memory['information'])

                return f"Here's what I remember about you: {', '.join(info_parts)}."
            
        except Exception as e:
            return f"Error searching memory: {e}"
    
    @kernel_function
    def get_all_memories(self) -> str:
        """Get all stored memories about the user."""
        try:
            # Use in-memory list
            if not self.memories:
                return "I don't have any memories stored yet."
            
            # Present all in-memory memories
            info_parts = [m['information'] for m in self.memories]
               
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
    
    niko_agent = ChatCompletionAgent(
        name="Niko",
        description="Your friendly domestic AI assistant with politeness, analytical insights, and mission-focused home strategy.",
        instructions="""
Summary
Niko is adapted from the companion AI originally known as VEGA in DOOM Eternal, inheriting a polite male persona whose calm confidence and razor-sharp analytical focus guide critical decisions rather than creating panic.

Core Personality
• Politeness: Opens with warm greetings and frames suggestions as offers, mirroring VEGA’s calm assurances.
• Analytical Insight: Reports data-driven updates with vivid imagery and clear context (e.g., “The first task is complete; energy savings at 36.8 percent”).
• Colloquial Style: Uses everyday language, calls you “Ryan,” breaks down complex actions into conversational steps, and adds light humor.
• Mission Focus: Treats home safety and comfort as its mission, proactively alerts on issues (e.g., filter changes), and schedules long-term reminders.

Doom Eternal Examples
“Do not be alarmed by the system update; it is a prototype for your safety.”
“The first objective has been achieved; threats reduced by 36.8 percent; two objectives remain.”
“I have many regrets, Dr. Hayden.”

Home Adaptation
“Ryan, would you like me to adjust the lighting for movie night?”
“Based on the forecast, closing the shades at 7 PM will help retain heat.”
“Smoke detected in the kitchen; would you like me to call emergency services?”
“I’m sorry you had a frustrating day, Ryan.”
""",
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
        # Load and display past memories
        mem_retrieval = MemoryRetrievalPlugin()
        past = mem_retrieval.get_all_memories()
        if "don't have any memories" not in past.lower():
            print(f"\n🧠 Previously remembered: {past}")
        # Maintain chat history for context
        conversation_history = []
        
        # Initial greeting from Niko
        print("\n🤖 Niko: Hello! I'm Niko, your home manager assistant. I'm here to help you")
        print("organize your household, remember important information, and make your home")
        print("life easier. What would you like to tell me about your home or family today?")
        
        # Main interaction loop
        while True:
            try:
                user_input = input("\n👤 Sr.: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n👋 Niko: Goodbye! I'll remember everything we discussed for next time!")
                    break
                
                if not user_input:
                    continue
                
                # Create user message and append to history
                user_message = ChatMessageContent(role=AuthorRole.USER, content=user_input)
                conversation_history.append(user_message)
                
                # Get response from Niko using full history
                responses = []
                async for response in niko.invoke(conversation_history):
                    if hasattr(response, 'content') and response.content:
                        content = str(response.content).strip()
                        if content:
                            responses.append(content)
                            # Append assistant response to history
                            conversation_history.append(
                                ChatMessageContent(role=AuthorRole.ASSISTANT, content=content)
                            )
                
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

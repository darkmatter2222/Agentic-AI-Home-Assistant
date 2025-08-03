import asyncio
import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.hugging_face import HuggingFaceTextCompletion
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
    
    print(f"🤖 Using OpenAI model: {model}")
    
    return OpenAIChatCompletion(**service_kwargs)


def create_local_granite_service():
    """Create local IBM Granite model service optimized for GPU."""
    try:
        # Try to import transformers for local model
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch
        
        # Detailed CUDA diagnostics
        print(f"🔍 PyTorch version: {torch.__version__}")
        print(f"🔍 CUDA compiled version: {torch.version.cuda}")
        print(f"🔍 CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"� CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"🔍 GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"� GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
        
        # Force CUDA usage if available, otherwise fallback to CPU
        # Special handling for RTX 5090 compatibility issues
        if torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16  # Use half precision for better GPU performance
            print(f"🎮 GPU DETECTED: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            # Check for RTX 5090 compatibility warning
            gpu_name = torch.cuda.get_device_name(0)
            if "RTX 5090" in gpu_name or "sm_120" in str(torch.cuda.get_device_capability(0)):
                print("⚠️  RTX 5090 detected with potential compatibility issues")
                print("💡 If GPU loading fails, will automatically fall back to CPU")
        else:
            device = "cpu"
            torch_dtype = torch.float32
            print("⚠️  GPU NOT AVAILABLE: Using CPU (will be slower)")
            print("💡 To enable GPU, install CUDA PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu124")
        
        print(f"🖥️  Initial device selection: {device}")
        
        model_name = "ibm-granite/granite-3.2-2b-instruct"
        print(f"🔄 Loading local model: {model_name}")
        print("⚠️  This may take a few minutes the first time...")
        
        # Load tokenizer
        print("📖 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Ensure tokenizer has a pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with GPU optimization and automatic CPU fallback
        print("🧠 Loading model...")
        
        if device == "cuda":
            # GPU loading with automatic CPU fallback for compatibility issues
            try:
                print("🚀 Attempting GPU loading with Flash Attention...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map="auto",  # Automatically map to GPU
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    attn_implementation="flash_attention_2"  # Try flash attention first
                )
                print("✅ Model loaded on GPU with Flash Attention 2")
                final_device = "cuda"
            except Exception as flash_error:
                print(f"⚠️  Flash Attention 2 failed: {flash_error}")
                try:
                    print("🔄 Attempting GPU loading with standard attention...")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch_dtype,
                        device_map="auto",  # Automatically map to GPU
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        attn_implementation="eager"  # Use standard attention
                    )
                    print("✅ Model loaded on GPU with standard attention")
                    final_device = "cuda"
                except Exception as gpu_error:
                    print(f"❌ GPU loading failed: {gpu_error}")
                    print("🔄 Falling back to CPU mode...")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,  # Use float32 for CPU
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        attn_implementation="eager"
                    )
                    print("✅ Model loaded on CPU")
                    final_device = "cpu"
        else:
            # CPU loading
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="eager"
            )
            print("✅ Model loaded on CPU")
            final_device = "cpu"
        
        # Create text generation pipeline with device-specific optimization
        print("🔧 Setting up generation pipeline...")
        text_generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=None,  # Prevent moving an accelerate-loaded model
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            batch_size=1,  # Optimize batch size for memory
            return_full_text=False
        )
        
        print(f"✅ Local model loaded successfully on {final_device.upper()}!")
        
        # Memory info after loading
        if final_device == "cuda":
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            memory_reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"📊 GPU memory allocated: {memory_allocated:.1f} GB")
            print(f"📊 GPU memory reserved: {memory_reserved:.1f} GB")
        else:
            print("💻 Running on CPU - responses will be slower but functional")
        
        return text_generator
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install with: pip install transformers torch accelerate safetensors")
        print("For GPU support: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        raise
    except Exception as e:
        print(f"❌ Error loading local model: {e}")
        raise


def get_model_choice():
    """Ask user to choose between OpenAI and local model."""
    print("\n🤖 Choose your AI model:")
    print("1. OpenAI (GPT-4) - Cloud-based, requires API key")
    print("2. Local Granite - IBM Granite 4.0 Tiny, runs locally")
    
    while True:
        choice = input("\nEnter your choice (1 or 2): ").strip()
        if choice == "1":
            return "openai"
        elif choice == "2":
            return "local"
        else:
            print("❌ Please enter 1 or 2")


def create_ai_service(model_type):
    """Create AI service based on chosen model type."""
    if model_type == "openai":
        return create_openai_service(), "openai"
    elif model_type == "local":
        return create_local_granite_service(), "local"
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_niko_agent_with_model(ai_service, model_type):
    """Create Niko agent with the specified AI service."""
    
    if model_type == "openai":
        # Use ChatCompletionAgent for OpenAI
        niko_agent = ChatCompletionAgent(
            name="Niko",
            description="Your friendly home manager who helps with household tasks, routines, and remembering important information.",
            instructions="""You are Niko, a friendly and helpful home manager assistant. 

Your main abilities:
- Have natural conversations about home and family
- Help with household organization and routines
- Automatically store and retrieve important information using your memory functions

MEMORY APPROACH:
You have three main memory functions:
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
            service=ai_service,
            plugins=[HomeAssistantPlugin(), MemoryStoragePlugin(), MemoryRetrievalPlugin()],
        )
        return niko_agent
    
    elif model_type == "local":
        # For local model, we'll create a GPU-optimized wrapper class
        class LocalNikoAgent:
            def __init__(self, text_generator):
                self.text_generator = text_generator
                self.memory_storage = MemoryStoragePlugin()
                self.memory_retrieval = MemoryRetrievalPlugin()
                self.home_assistant = HomeAssistantPlugin()
                self.name = "Niko"
                
                # Import torch for GPU memory management
                try:
                    import torch
                    self.torch = torch
                    self.has_torch = True
                except ImportError:
                    self.torch = None
                    self.has_torch = False
            
            async def invoke(self, messages):
                # Extract user message
                user_message = messages[0].content if messages else ""
                
                # Enhanced prompt engineering for the local model
                system_prompt = """You are Niko, a friendly home manager assistant. You help with household tasks and remember important information.

Key behaviors:
- Be conversational and helpful
- When users tell you personal info, acknowledge it warmly
- Keep responses concise but friendly
- Focus on home management topics

User: {user_input}
Niko:"""
                
                # Create full prompt
                full_prompt = system_prompt.format(user_input=user_message)
                
                try:
                    # Clear GPU cache before generation if available
                    if self.has_torch and self.torch.cuda.is_available():
                        self.torch.cuda.empty_cache()
                    
                    # Generate response with optimized parameters
                    response_data = self.text_generator(
                        full_prompt,
                        max_new_tokens=150,  # Reduced for faster generation
                        temperature=0.7,
                        top_p=0.9,
                        top_k=50,
                        do_sample=True,
                        return_full_text=False,
                        clean_up_tokenization_spaces=True
                    )
                    
                    # Extract and clean response
                    response = response_data[0]['generated_text'] if response_data else ""
                    response = response.strip()
                    
                    # Remove any prompt artifacts
                    if response.startswith("Niko:"):
                        response = response[5:].strip()
                    if response.startswith("User:"):
                        response = response.split("Niko:")[-1].strip()
                    
                    # Clean up common generation artifacts
                    response = response.replace("\n\n", " ").replace("  ", " ").strip()
                    
                    # Ensure response isn't empty
                    if not response:
                        response = "I'm here to help! Could you tell me more about what you need?"
                    
                    # Check if user mentioned storing information and handle memory
                    if any(keyword in user_message.lower() for keyword in 
                           ["my name", "i have", "i am", "i live", "my wife", "my husband", "my kids", "my children"]):
                        try:
                            memory_result = self.memory_storage.store_memory(user_message, "conversation")
                            # Don't add memory confirmation to response as it's already printed
                        except Exception as e:
                            print(f"⚠️  Memory storage failed: {e}")
                    
                    # Handle memory retrieval requests
                    if any(keyword in user_message.lower() for keyword in 
                           ["who am i", "tell me about", "what do you know", "about my life", "my family"]):
                        try:
                            memory_info = self.memory_retrieval.get_all_memories()
                            if "don't have any memories" not in memory_info:
                                response = f"{response} {memory_info}"
                        except Exception as e:
                            print(f"⚠️  Memory retrieval failed: {e}")
                    
                    # Clear GPU cache after generation if available
                    if self.has_torch and self.torch.cuda.is_available():
                        self.torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"⚠️  Generation error: {e}")
                    response = "I'm having trouble processing that right now. Could you try rephrasing?"
                
                # Return response in expected format as async generator
                class MockResponse:
                    def __init__(self, content):
                        self.content = content
                
                # Yield the response to make it an async generator
                yield MockResponse(response)
        
        return LocalNikoAgent(ai_service)


async def main():
    """Main function to run Niko's Home Assistant with model selection."""
    print("🏠 Starting Niko's Home Assistant (Multi-Model)")
    print("=" * 60)
    
    # Get model choice
    model_choice = get_model_choice()
    
    print("\n" + "=" * 60)
    print("👋 Meet Niko - Your friendly home manager")
    print("💾 Niko can store and remember important information")
    print("🔍 Niko can search through your stored memories")
    print("=" * 60)
    print("Tell Niko about your household, routines, preferences, and important information.")
    print("Niko will remember everything and help you manage your home!")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.")
    print("=" * 60)
    
    try:
        # Create AI service and Niko agent
        ai_service, model_type = create_ai_service(model_choice)
        niko = create_niko_agent_with_model(ai_service, model_type)
        
        # Initial greeting from Niko
        print(f"\n🤖 Niko ({model_type.upper()}): Hello! I'm Niko, your home manager assistant. I'm here to help you")
        print("organize your household, remember important information, and make your home")
        print("life easier. What would you like to tell me about your home or family today?")
        
        # Main interaction loop
        while True:
            try:
                user_input = input("\n👤 Sr.: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print(f"\n👋 Niko ({model_type.upper()}): Goodbye! I'll remember everything we discussed for next time!")
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
                        print(f"\n🤖 Niko ({model_type.upper()}): {response}")
                
            except KeyboardInterrupt:
                print(f"\n\n👋 Niko ({model_type.upper()}): Goodbye! Take care!")
                break
            except Exception as e:
                print(f"\n❌ Error during conversation: {e}")
                print("Please try again or type 'quit' to exit.")
        
    except Exception as e:
        print(f"❌ Error starting Niko's Home Assistant: {e}")
        if model_choice == "openai":
            print("\nPlease check your .env file and ensure your OpenAI credentials are correct.")
        else:
            print("\nPlease ensure you have the required dependencies installed:")
            print("pip install transformers torch")


if __name__ == "__main__":
    asyncio.run(main())

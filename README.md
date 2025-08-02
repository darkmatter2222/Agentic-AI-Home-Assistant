# Niko's AI Home Assistant

Meet **Niko**, your friendly AI home manager! This intelligent home assistant system uses Microsoft Semantic Kernel and OpenAI to help you organize your household, remember important information, and manage your home life efficiently.

## 🏠 Meet Your Home Team

### 🤖 **Niko - Your Home Manager**
- Friendly and personable home assistant
- Helps with household routines, schedules, and organization
- Remembers family preferences and important information
- Coordinates with the memory system to store and retrieve information

### 📝 **Memory Agent - Information Storage**
- Automatically stores important household information
- Organizes details into categories (family, schedule, preferences, etc.)
- Creates searchable long-term memory storage

### 🔍 **Memory Retrieval Agent - Information Search**
- Finds stored information when needed
- Searches through memories using natural language
- Provides comprehensive results from memory searches

## ✨ Key Features

- **Intelligent Home Management**: Niko helps organize your household and daily routines
- **Long-Term Memory**: Automatically stores and retrieves important family and household information
- **File-Based Storage**: Memories are saved locally in `home_memories.json` for persistence
- **Smart Categorization**: Information is organized by type (family, schedule, preferences, etc.)
- **Natural Conversation**: Talk to Niko naturally about your home and family
- **Context Awareness**: Maintains conversation context and references previous discussions
- **Console Interface**: Easy-to-use command-line interface

## 🚀 Quick Start

1. **Configure your OpenAI credentials** in the `.env` file:
   ```
   OPENAI_API_KEY=your_actual_openai_api_key_here
   OPENAI_MODEL=gpt-4
   ```

2. **Run the application**:
   ```powershell
   .\run.ps1
   ```
   
   Or directly:
   ```powershell
   python main.py
   ```

3. **Start talking to Niko!**

## 💬 Example Conversations

### Getting Started
- **"Hi Niko! Let me tell you about my family..."**
- **"My wife Sarah loves coffee every morning at 7 AM"**
- **"My son Jake is 8 years old and plays soccer on Tuesdays"**

### Home Management
- **"Remember that we need to change the air filter every 3 months"**
- **"Set up a grocery list - we always need milk, bread, and eggs"**
- **"My mother-in-law visits every second Sunday"**

### Retrieving Information
- **"What do you remember about Jake's schedule?"**
- **"When does Sarah usually have her coffee?"**
- **"What household maintenance do we need to do?"**

## 🧠 Memory Categories

Niko automatically organizes information into categories:

- **family** - Family member details, preferences, schedules
- **schedule** - Routines, recurring events, appointments
- **preferences** - Likes, dislikes, habits
- **home** - Maintenance, layout, equipment information
- **events** - Important dates, celebrations, special occasions
- **health** - Medical info, dietary needs, medications
- **contacts** - Important people, service providers

## 🔧 Technical Details

### Memory Storage
- Information is stored in `home_memories.json` in the application directory
- Each memory includes: key, information, category, timestamp, and unique ID
- Memories persist between sessions

### Agent Architecture
- **Handoff Orchestration**: Agents intelligently pass conversations to specialists
- **Context Preservation**: Full conversation history is maintained across handoffs
- **Smart Routing**: Niko decides when to store information or retrieve memories

### Requirements
- Python 3.8 or higher
- OpenAI API key
- Required packages: `semantic-kernel`, `python-dotenv`, `openai`

## 📁 Project Structure

```
├── main.py                 # Main application with Niko and memory agents
├── .env                    # OpenAI configuration
├── requirements.txt        # Python dependencies
├── home_memories.json      # Stored memories (auto-created)
├── run.ps1                # PowerShell run script
├── test_niko_memory.py     # Memory functionality test
└── README.md              # This file
```

## 🛠️ Development & Testing

Test the memory system:
```bash
python test_niko_memory.py
```

The test will:
1. Tell Niko about family information
2. Ask Niko to recall the information
3. Verify that the memory system is working

## 🔒 Privacy & Security

- All memories are stored locally on your machine
- No data is sent to external services except OpenAI for processing
- You have full control over your stored information
- Memory file can be backed up or deleted as needed

## 📞 Support

If you encounter issues:
1. Verify your OpenAI API key is correct in `.env`
2. Check that all dependencies are installed: `pip install -r requirements.txt`
3. Ensure you have Python 3.8+ installed
4. Check the console output for detailed error messages

## 🎯 Usage Tips

- **Be specific**: The more details you give Niko, the better it can help you
- **Use categories**: Mention if something is about family, home maintenance, schedules, etc.
- **Ask for summaries**: "What do you remember about..." to get stored information
- **Update information**: Tell Niko when things change, and it will update the memories

---

Welcome to your new AI-powered home management system with Niko! 🏠🤖

A multi-agent AI system built with Microsoft Semantic Kernel and OpenAI's o1 model. This application demonstrates agent orchestration with handoff capabilities between three specialized agents: Manager, Developer, and Project Manager.

## Key Features

- **Manager Agent**: Provides strategic direction, coordinates teams, and handles high-level decisions
- **Developer Agent**: Handles technical specifications, code reviews, and development estimates
- **Project Manager Agent**: Manages timelines, resources, and project coordination
- **Dynamic Handoff**: Agents can intelligently transfer conversations to the most appropriate specialist
- **Conversation Memory**: Agents maintain context across the entire conversation and can reference previous discussions
- **Console Interface**: Interactive command-line interface for seamless communication

## Prerequisites

- Python 3.8 or higher
- OpenAI API key with access to o1-preview model
- Git (for cloning the repository)

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Agentic-AI-Home-Assistant
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   - Copy the `.env` file and update it with your OpenAI credentials:
   ```
   OPENAI_API_KEY=your_actual_api_key_here
   OPENAI_MODEL=o1-preview
   OPENAI_ORG_ID=your_organization_id_here  # Optional
   ```

4. **Run the application**:
   ```bash
   python main.py
   ```

## Usage Examples

Once the application is running, you can interact with the agents using natural language. Here are some example interactions:

### Project Management
- "I need to create a timeline for a new mobile app project with a team of 5 developers"
- "Track the progress of project ABC-123"
- "Allocate 3 senior developers to the authentication feature"

### Development Tasks
- "Estimate the development time for implementing user authentication with OAuth"
- "Create a technical specification for a real-time chat feature"
- "Review the performance of our database queries"

### Strategic Decisions
- "What should be our priority for the next quarter's development?"
- "I need help deciding between microservices and monolithic architecture"
- "Create a high-level task for implementing AI features"

## Agent Capabilities

### Manager Agent
- Strategic planning and decision making
- Resource allocation coordination
- Team coordination and task management
- Business objective alignment

### Developer Agent
- Technical specification creation
- Development time estimation
- Code review and quality assessment
- Architecture recommendations

### Project Manager Agent
- Project timeline creation and management
- Progress tracking and reporting
- Resource allocation and coordination
- Milestone and dependency management

## How It Works

The system uses Microsoft Semantic Kernel's agent orchestration with handoff capabilities and automatic memory management. When you submit a request:

1. **Initial Processing**: The Manager Agent initially receives your request
2. **Context Awareness**: All agents have access to the complete conversation history
3. **Smart Routing**: Based on the content, agents either handle requests directly or hand off to appropriate specialists
4. **Memory Preservation**: Each handoff maintains the conversation context, so agents can reference previous discussions
5. **Specialized Functions**: Each agent has access to specialized plugins with relevant functions
6. **Continuous Context**: The conversation continues with full memory until task completion

### Memory Management

The **HandoffOrchestration** automatically manages conversation memory through:

- **Shared Conversation History**: All agents can access and reference the complete conversation
- **Context Preservation**: When agents hand off to each other, the full context is maintained
- **Cross-Agent Memory**: An agent can reference decisions made by other agents earlier in the conversation
- **Persistent Context**: Information from early in the conversation remains available throughout

### Example Memory Usage

1. **User**: "I want to build a calculator app"
2. **Manager**: Discusses project scope and requirements
3. **User**: "For the calculator we discussed, what technology should I use?"
4. **Developer**: References the calculator project from step 1 and provides technical recommendations
5. **User**: "Create a timeline for the calculator project with that technology"
6. **Project Manager**: References both the calculator project and technology choice to create timeline

## Technical Architecture

- **Semantic Kernel**: Microsoft's AI orchestration framework
- **OpenAI o1-preview**: Advanced reasoning model for complex problem solving
- **Agent Plugins**: Modular functions for specific domain tasks
- **Handoff Orchestration**: Dynamic agent routing based on context

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_MODEL`: Model to use (default: o1-preview)
- `OPENAI_ORG_ID`: Your OpenAI organization ID (optional)

### Supported Models
While configured for o1-preview, the system can work with other OpenAI models:
- gpt-4
- gpt-4-turbo
- gpt-3.5-turbo

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **API Key Issues**: Verify your OpenAI API key is correct and has proper permissions

3. **Model Access**: Ensure your account has access to the o1-preview model

4. **Rate Limits**: The o1 models have lower rate limits - if you encounter limits, consider switching to gpt-4

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

See LICENSE file for details.

## Support

If you encounter issues:
1. Check the troubleshooting section
2. Verify your environment configuration
3. Check the console output for detailed error messages
4. Ensure you have the latest version of dependencies
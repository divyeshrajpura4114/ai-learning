# Hotel Finder Agent

An intelligent AI-powered hotel recommendation system that helps users find the best hotels in any city. The agent uses a goal-oriented architecture with function calling to geocode locations, search for hotels, and provide ranked recommendations based on user preferences.

## Features

- **Intelligent Geocoding**: Automatically converts city names to geographic coordinates using the Nominatim API
- **Hotel Search**: Finds hotels within a specified radius using OpenStreetMap data via the Overpass API
- **Smart Recommendations**: Analyzes hotels based on location, price tier, pet-friendliness, star rating, and other preferences
- **AI-Powered**: Uses Large Language Models (LLMs) with tool-calling capabilities for natural interaction
- **Extensible Architecture**: Modular design with goals, actions, memory, and environment components

## Architecture

The project follows a goal-oriented agent architecture:

- **Goals**: Define the agent's objectives (Geocode Location, Find Hotels, Rank and Recommend)
- **Actions**: Callable functions for specific tasks (geocode_city, search_hotels, terminate)
- **Memory**: Stores conversation history and results
- **Environment**: Executes actions and handles results
- **Agent Language**: Formats prompts and parses LLM responses for tool calling

## Prerequisites

- Python 3.8+
- API keys for LLM service (Groq recommended)
- Internet connection for API calls

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd hotel-finder-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

Required environment variables:
- `GROQ_API_KEY`: Your Groq API key for LLM access

## Usage

Run the hotel finder agent:

```bash
python3 main.py
```

When prompted, enter your city and preferences:

```
Describe what you're looking for (city + preferences): Kolkata luxury 5-star pet-friendly
```

The agent will:
1. Geocode the city to get coordinates
2. Search for hotels within 5km radius
3. Rank and recommend the top 5 matches based on your criteria

## Configuration

### LLM Model

The agent uses Groq's Llama models by default. You can modify the model in `main.py`:

```python
model="groq/llama-3.1-70b-versatile"
```

### Search Parameters

- Default search radius: 5km (configurable in the search_hotels action)
- APIs used: Nominatim (geocoding), Overpass (hotel data)

## Project Structure

```
hotel-finder-agent/
├── main.py                 # Main entry point and action definitions
├── agent.py                # Agent implementation and language classes
├── base/                   # Core agent components
│   ├── __init__.py
│   ├── action.py          # Action and ActionRegistry classes
│   ├── agent_language.py  # Base language interface
│   ├── environment.py     # Execution environment
│   ├── goal.py            # Goal definition
│   ├── memory.py          # Memory management
│   └── prompt.py          # Prompt construction
├── .env                    # Environment variables (API keys)
├── README.md              # This file
└── requirements.txt       # Python dependencies
```

## Dependencies

- `litellm`: For LLM API calls
- `requests`: For HTTP API requests
- `python-dotenv`: For environment variable loading
- `dataclasses`: For data structures (Python 3.7+ built-in)

## API Usage

The agent uses free, open APIs:

- **Nominatim**: For geocoding city names to coordinates
- **Overpass API**: For querying OpenStreetMap data to find hotels

No API keys required for geocoding and hotel search - they use free public services.

## Customization

### Adding New Actions

1. Define your function in `main.py`
2. Register it with the ActionRegistry
3. Add appropriate goals if needed

### Modifying Goals

Edit the goals list in `main.py` to change the agent's behavior and objectives.

### Extending the Agent

The modular architecture allows for easy extension:
- Add new agent languages for different LLM formats
- Implement custom memory systems
- Create new environment types

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure `GROQ_API_KEY` is set in your `.env` file
2. **Network Errors**: Check internet connection for API calls
3. **No Hotels Found**: Try a larger search radius or different city
4. **LLM Not Responding**: Verify your API key and model availability

### Debug Mode

Add print statements in `main.py` to see prompts and responses:

```python
print("PROMPT", prompt.messages)
print("RESPONSE", response)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source. Please check the license file for details.

## Acknowledgments

- OpenStreetMap and Nominatim for geospatial data
- Groq for LLM API access
- The AI agent architecture inspired by various research in autonomous agents
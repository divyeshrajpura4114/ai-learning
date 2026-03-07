import os
import json
import requests
from dataclasses import dataclass, field
from typing import List, Callable, Any, Dict

from base.goal import Goal
from base.action import Action, ActionRegistry
from base.memory import Memory
from base.environment import Environment
from base.prompt import Prompt
from base.agent_language import AgentLanguage
from agent import Agent, AgentFunctionCallingActionLanguage

from litellm import completion
from dotenv import load_dotenv

load_dotenv()
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL  = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "GAME-HotelAgent/1.0", "Accept-Language": "en"}

def generate_response(prompt: Prompt) -> str:
    """Call LLM to get response"""

    messages = prompt.messages
    tools = prompt.tools

    result = None

    if not tools:
        response = completion(
            model="groq/openai/gpt-oss-20b",
            messages=messages,
            max_tokens=1024,
            api_key = os.environ.get("GROQ_API_KEY")
        )
        result = response.choices[0].message.content
    else:
        response = completion(
            model="groq/openai/gpt-oss-20b",
            messages=messages,
            tools=tools,
            max_tokens=1024,
            api_key = os.environ.get("GROQ_API_KEY")
        )

        if response.choices[0].message.tool_calls:
            tool = response.choices[0].message.tool_calls[0]
            result = {
                "tool": tool.function.name,
                "args": json.loads(tool.function.arguments),
            }
            result = json.dumps(result)
        else:
            result = response.choices[0].message.content


    return result

def geocode_city(city: str) -> Dict:
    """Resolve a city name to lat/lon via Nominatim (free, no API key)."""
    resp = requests.get(
        NOMINATIM_URL,
        params={"q": city, "format": "json", "limit": 1},
        headers=HEADERS,
        timeout=10,
    )
    results = resp.json()
    if not results:
        return {"error": f"City '{city}' not found"}
    r = results[0]
    return {
        "city": r["display_name"],
        "lat": float(r["lat"]),
        "lon": float(r["lon"]),
    }


def search_hotels(lat: float, lon: float, radius_km: float = 5) -> List[Dict]:
    """Fetch hotels from OpenStreetMap via Overpass API (free, no API key)."""
    radius_m = int(radius_km * 1000)
    query = f"""
[out:json][timeout:30];
(
  node["tourism"="hotel"](around:{radius_m},{lat},{lon});
  way["tourism"="hotel"](around:{radius_m},{lat},{lon});
  node["tourism"="motel"](around:{radius_m},{lat},{lon});
  node["tourism"="guest_house"](around:{radius_m},{lat},{lon});
  node["tourism"="hostel"](around:{radius_m},{lat},{lon});
);
out body center 60;
""".strip()

    resp = requests.post(OVERPASS_URL, data=query, timeout=30)
    hotels = []
    for el in resp.json().get("elements", []):
        tags = el.get("tags", {})
        if not tags.get("name"):
            continue
        hotels.append({
            "name":       tags.get("name"),
            "type":       tags.get("tourism", "hotel"),
            "stars":      tags.get("stars") or tags.get("stars:official"),
            "pets":       tags.get("pets") or tags.get("dog") or tags.get("pets_allowed"),
            "wifi":       tags.get("internet_access"),
            "wheelchair": tags.get("wheelchair"),
            "website":    tags.get("website") or tags.get("contact:website"),
            "address":    " ".join(filter(None, [
                              tags.get("addr:housenumber"),
                              tags.get("addr:street"),
                              tags.get("addr:city"),
                          ])),
        })
    return hotels


def terminate(message: str) -> str:
    """Deliver the final hotel recommendations to the user."""
    return message

def main():
    # Define the agent's goals
    goals = [
        Goal(priority = 1,
             name = "Geocode Location",
             description = "Convert the user's city name into geographic coordinates using the geocode_city action before doing anything else."),
        Goal(priority = 2,
             name = "Find Hotels",
             description = "Search for hotels near the geocoded coordinates using search_hotels. Use a radius of 3-5km for city centres, larger for rural areas."),
        Goal(priority = 3,
             name = "Rank and Recommend",
             description = "Analyse the retrieved hotels against the user's criteria — location, price tier, pet friendliness, star rating, and any other preferences. Call terminate with a ranked list of exactly the top 5 matches and a brief explanation for each recommendation."),
    ]

    # Define the agent's language
    agent_language = AgentFunctionCallingActionLanguage()

# ohealth
    # Define the action registry and register some actions
    action_registry = ActionRegistry()
    action_registry.register(Action(name = "search_hotels",
                                    function = search_hotels,
                                    description = "Searches for hotels near a lat/lon coordinate using the Overpass API (OpenStreetMap data). Call geocode_city first to get coordinates. Returns a list of hotels with name, star rating, pet policy, address, and type (hotel/motel/guest_house). radius_km controls the search area — use 3 for dense cities, 10+ for rural areas.",
                                    parameters={"type": "object",
                                                "properties": {
                                                    "lat": {"type": "number", "description": "Latitude from geocode_city"},
                                                    "lon": {"type": "number", "description": "Longitude from geocode_city"},
                                                    "radius_km": {"type": "number", "description": "Search radius in km (default 5)"},
                                                },
                                                "required": ["lat", "lon"]
                                    },
                                    terminal = False
                                ))
    action_registry.register(Action(name = "geocode_city",
                                    function = geocode_city,
                                    description = "Converts a city name into geographic coordinates (lat/lon) using the Nominatim API. Always call this before search_hotels.",
                                    parameters = {"type": "object",
                                                    "properties": {
                                                        "city": {
                                                            "type": "string",
                                                            "description": "City name, e.g. 'Paris' or 'Tokyo, Japan'"
                                                        }
                                                    },
                                                    "required": ["city"]
                                                },
                                    terminal = False
                                ))
    action_registry.register(Action(name ="terminate",
                                    function =terminate,
                                    description = "Ends the session and delivers the final answer to the user. Call this after ranking hotels. The message should contain a numbered list of the top 5 hotels with a brief explanation for each, covering how well it matches the user's criteria.",
                                    parameters = {"type": "object",
                                                    "properties": {
                                                        "message": {
                                                            "type": "string",
                                                            "description": "The final hotel recommendations"
                                                        }
                                                    },
                                                    "required": ["message"]
                                                },
                                    terminal = True,
                                ))

    # Define the environment
    environment = Environment()

    # Create an agent instance
    agent = Agent(goals, agent_language, action_registry, generate_response, environment)

    # Run the agent with user input
    user_input = input("Describe your criteria such as location, price tier, pet friendliness, star rating, and any other preferences for findig best hotels):")

    final_memory = agent.run(user_input, max_iterations = 5)

    # Print the final memory
    print(final_memory.get_memories())

if __name__ == "__main__":
    main()

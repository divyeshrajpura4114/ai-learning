import os
import json
import inspect
import requests
from dataclasses import dataclass, field
from typing import List, Callable, Any, Dict, get_type_hints

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

tools = {}
tools_by_tag = {}

def get_tool_metadata(func,
                      tool_name = None,
                      description = None,
                      parameters_override = None,
                      terminal = False,
                      tags = None):
    """
    Extracts metadata for a function to use in tool registration.

    Parameters:
        func (function): The function to extract metadata from.
        tool_name (str, optional): The name of the tool. Defaults to the function name.
        description (str, optional): Description of the tool. Defaults to the function's docstring.
        parameters_override (dict, optional): Override for the argument schema. Defaults to dynamically inferred schema.
        terminal (bool, optional): Whether the tool is terminal. Defaults to False.
        tags (List[str], optional): List of tags to associate with the tool.

    Returns:
        dict: A dictionary containing metadata about the tool, including description, args schema, and the function.
    """
    # Default tool_name to the function name if not provided
    tool_name = tool_name or func.__name__

    # Default description to the function's docstring if not provided
    description = description or (func.__doc__.strip() if func.__doc__ else "No description provided.")

    # Discover the function's signature and type hints if no args_override is provided
    if parameters_override is None:
        signature = inspect.signature(func)
        type_hints = get_type_hints(func)

        # Build the arguments schema dynamically
        args_schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        for param_name, param in signature.parameters.items():

            if param_name in ["action_context", "action_agent"]:
                continue  # Skip these parameters

            def get_json_type(param_type):
                if param_type == str:
                    return "string"
                elif param_type == int:
                    return "integer"
                elif param_type == float:
                    return "number"
                elif param_type == bool:
                    return "boolean"
                elif param_type == list:
                    return "array"
                elif param_type == dict:
                    return "object"
                else:
                    return "string"

            # Add parameter details
            param_type = type_hints.get(param_name, str)  # Default to string if type is not annotated
            param_schema = {"type": get_json_type(param_type)}  # Convert Python types to JSON schema types

            args_schema["properties"][param_name] = param_schema

            # Add to required if not defaulted
            if param.default == inspect.Parameter.empty:
                args_schema["required"].append(param_name)
    else:
        args_schema = parameters_override

    # Return the metadata as a dictionary
    return {
        "tool_name": tool_name,
        "description": description,
        "parameters": args_schema,
        "function": func,
        "terminal": terminal,
        "tags": tags or []
    }

def register_tool(tool_name = None,
                  description = None,
                  parameters_override = None,
                  terminal = False,
                  tags = None
                ):
    """
    A decorator to dynamically register a function in the tools dictionary with its parameters, schema, and docstring.

    Parameters:
        tool_name (str, optional): The name of the tool to register. Defaults to the function name.
        description (str, optional): Override for the tool's description. Defaults to the function's docstring.
        parameters_override (dict, optional): Override for the argument schema. Defaults to dynamically inferred schema.
        terminal (bool, optional): Whether the tool is terminal. Defaults to False.
        tags (List[str], optional): List of tags to associate with the tool.

    Returns:
        function: The wrapped function.
    """
    def decorator(func):
        # Use the reusable function to extract metadata
        metadata = get_tool_metadata(func = func,
                                        tool_name = tool_name,
                                        description = description,
                                        parameters_override = parameters_override,
                                        terminal = terminal,
                                        tags = tags
                                    )

        # Register the tool in the global dictionary
        tools[metadata["tool_name"]] = {"description": metadata["description"],
                                        "parameters": metadata["parameters"],
                                        "function": metadata["function"],
                                        "terminal": metadata["terminal"],
                                        "tags": metadata["tags"] or []
                                    }

        for tag in metadata["tags"]:
            if tag not in tools_by_tag:
                tools_by_tag[tag] = []
            tools_by_tag[tag].append(metadata["tool_name"])

        return func
    return decorator

class PythonActionRegistry(ActionRegistry):
    def __init__(self, tags: List[str] = None, tool_names: List[str] = None):
        super().__init__()

        self.terminate_tool = None

        for tool_name, tool_desc in tools.items():
            if tool_name == "terminate":
                self.terminate_tool = tool_desc

            if tool_names and tool_name not in tool_names:
                continue

            tool_tags = tool_desc.get("tags", [])
            if tags and not any(tag in tool_tags for tag in tags):
                continue

            self.register(Action(name = tool_name,
                                function = tool_desc["function"],
                                description = tool_desc["description"],
                                parameters = tool_desc.get("parameters", {}),
                                terminal = tool_desc.get("terminal", False)
                            ))

    def register_terminate_tool(self):
        if self.terminate_tool:
            self.register(Action(name = "terminate",
                                function = self.terminate_tool["function"],
                                description = self.terminate_tool["description"],
                                parameters = self.terminate_tool.get("parameters", {}),
                                terminal = self.terminate_tool.get("terminal", False)
                            ))
        else:
            raise Exception("Terminate tool not found in tool registry")

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

@register_tool(tags = ["geocode", "city"])
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

@register_tool(tags = ["search", "hotesl"])
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

@register_tool(tags = ["system"], terminal = True)
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

    # Define the action registry and register some actions
    action_registry = PythonActionRegistry(tags=["geocode", "search", "hotels", "system"])

    # Define the environment
    environment = Environment()

    # Create an agent instance
    agent = Agent(goals = goals,
                    agent_language = agent_language,
                    action_registry = action_registry,
                    generate_response = generate_response,
                    environment = environment
                )

    # Run the agent with user input
    user_input = input("Describe your criteria such as location, price tier, pet friendliness, star rating, and any other preferences for findig best hotels):")

    final_memory = agent.run(user_input, max_iterations = 5)

    # Print the final memory
    print(final_memory.get_memories())

if __name__ == "__main__":
    main()

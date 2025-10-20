import json
import os

import azure.identity
import openai
from dotenv import load_dotenv

# Setup the OpenAI client to use either Azure, OpenAI.com, or Ollama API
load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "github")

if API_HOST == "azure":
    token_provider = azure.identity.get_bearer_token_provider(
        azure.identity.DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    client = openai.OpenAI(
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=token_provider,
    )
    MODEL_NAME = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]

elif API_HOST == "ollama":
    client = openai.OpenAI(base_url=os.environ["OLLAMA_ENDPOINT"], api_key="nokeyneeded")
    MODEL_NAME = os.environ["OLLAMA_MODEL"]

elif API_HOST == "github":
    client = openai.OpenAI(base_url="https://models.github.ai/inference", api_key=os.environ["GITHUB_TOKEN"])
    MODEL_NAME = os.getenv("GITHUB_MODEL", "openai/gpt-4o")

else:
    client = openai.OpenAI(api_key=os.environ["OPENAI_KEY"])
    MODEL_NAME = os.environ["OPENAI_MODEL"]


tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": "Lookup the weather for a given city name or zip code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_name": {
                        "type": "string",
                        "description": "The city name",
                    },
                    "zip_code": {
                        "type": "string",
                        "description": "The zip code",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_movies",
            "description": "Lookup movies playing in a given city name or zip code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_name": {
                        "type": "string",
                        "description": "The city name",
                    },
                    "zip_code": {
                        "type": "string",
                        "description": "The zip code",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool (function) implementations
# ---------------------------------------------------------------------------
def lookup_weather(city_name: str | None = None, zip_code: str | None = None) -> str:
    """Looks up the weather for given city_name and zip_code."""
    location = city_name or zip_code or "unknown"
    # In a real implementation, call an external weather API here.
    return {
        "location": location,
        "condition": "rain showers",
        "rain_mm_last_24h": 7,
        "recommendation": "Good day for indoor activities if you dislike drizzle.",
    }


def lookup_movies(city_name: str | None = None, zip_code: str | None = None) -> str:
    """Returns a list of movies playing in the given location."""
    location = city_name or zip_code or "unknown"
    # A real implementation could query a cinema listings API.
    return {
        "location": location,
        "movies": [
            {"title": "The Quantum Reef", "rating": "PG-13"},
            {"title": "Storm Over Harbour Bay", "rating": "PG"},
            {"title": "Midnight Koala", "rating": "R"},
        ],
    }


tool_mapping = {
    "lookup_weather": lookup_weather,
    "lookup_movies": lookup_movies,
}


# ---------------------------------------------------------------------------
# Conversation loop
# ---------------------------------------------------------------------------
messages = [
    {"role": "system", "content": "You are a tourism chatbot."},
    {"role": "user", "content": "Is it rainy enough in Sydney to watch movies and which ones are on?"},
]

print(f"Model: {MODEL_NAME} on Host: {API_HOST}\n")

while True:
    print("Calling model...\n")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,  # includes prior tool outputs
        tools=tools,
        tool_choice="auto",
        parallel_tool_calls=False,  # ensure sequential tool calls
    )

    assistant_message = response.choices[0].message
    # If the assistant returned standard content with no tool calls, we're done.
    if not assistant_message.tool_calls:
        print("Assistant:")
        print(assistant_message.content)
        break

    # Append the assistant tool request message to conversation
    messages.append(
        {
            "role": "assistant",
            "content": assistant_message.content or "",
            "tool_calls": [tc.model_dump() for tc in assistant_message.tool_calls],
        }
    )

    # Execute each requested tool sequentially.
    for tool_call in assistant_message.tool_calls:
        fn_name = tool_call.function.name
        raw_args = tool_call.function.arguments or "{}"
        print(f"Tool request: {fn_name}({raw_args})")
        target_tool = tool_mapping.get(fn_name)
        parsed_args = json.loads(raw_args)
        tool_result = target_tool(**parsed_args)
        tool_result_str = json.dumps(tool_result)
        # Provide the tool output back to the model
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": fn_name,
                "content": tool_result_str,
            }
        )

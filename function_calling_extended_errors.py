"""Extended function calling example with robust error handling.

This script is similar to the simple extended example but adds:
 - Safe JSON argument parsing (malformed JSON won't crash loop)
 - Tool existence validation (graceful error if model asks for unknown tool)
 - Exception safety around tool execution
 - JSON serialization of tool outputs for model consumption

It demonstrates a single round-trip (not a while loop) but with the
same safeguards implemented in `function_calling_while_loop.py`.
"""

import json
import os
from collections.abc import Callable
from typing import Any

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


# ---------------------------------------------------------------------------
# Tool implementation(s)
# ---------------------------------------------------------------------------
def lookup_weather(city_name: str | None = None, zip_code: str | None = None) -> dict[str, Any]:
    """Lookup the weather for a given city name or zip code.

    Returns a simple deterministic stub so the focus is on tool call flow.
    """
    location = city_name or zip_code or "unknown"
    return {
        "location": location,
        "weather": "sunny",
        "temperature_f": 75,
        "advice": "Great day to be outside!",
    }


tool_mapping: dict[str, Callable[..., Any]] = {
    "lookup_weather": lookup_weather,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": "Lookup the weather for a given city name or zip code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_name": {"type": "string", "description": "The city name"},
                    "zip_code": {"type": "string", "description": "The zip code"},
                },
                "strict": True,
                "additionalProperties": False,
            },
        },
    }
]


messages: list[dict[str, Any]] = [
    {"role": "system", "content": "You are a weather chatbot."},
    {"role": "user", "content": "is it sunny in berkeley CA?"},
]

print(f"Model: {MODEL_NAME} on Host: {API_HOST}\n")

# First model response (may include tool call)
response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages,
    tools=tools,
    tool_choice="auto",
    parallel_tool_calls=False,
)

assistant_msg = response.choices[0].message

# If no tool calls were requested, just print the answer.
if not assistant_msg.tool_calls:
    print("Assistant:")
    print(assistant_msg.content)
else:
    # Append assistant message including tool call metadata
    messages.append(
        {
            "role": "assistant",
            "content": assistant_msg.content or "",
            "tool_calls": [tc.model_dump() for tc in assistant_msg.tool_calls],
        }
    )

    # Process each requested tool sequentially (though usually one here)
    for tool_call in assistant_msg.tool_calls:
        fn_name = tool_call.function.name
        raw_args = tool_call.function.arguments or "{}"
        print(f"Tool request: {fn_name}({raw_args})")

        target = tool_mapping.get(fn_name)
        if not target:
            tool_result: Any = f"ERROR: No implementation registered for tool '{fn_name}'"
        else:
            # Parse arguments safely
            try:
                parsed_args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                parsed_args = {}
                tool_result = "Warning: Malformed JSON arguments received; proceeding with empty args"
            else:
                try:
                    tool_result = target(**parsed_args)
                except Exception as e:  # safeguard tool execution
                    tool_result = f"Tool execution error in {fn_name}: {e}"

        # Serialize tool output (dict or str) as JSON string for the model
        try:
            tool_content = json.dumps(tool_result)
        except Exception:
            # Fallback to string conversion if something isn't JSON serializable
            tool_content = json.dumps({"result": str(tool_result)})

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": fn_name,
                "content": tool_content,
            }
        )

    # Follow-up model response after supplying tool outputs
    followup = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tools,
    )
    final_msg = followup.choices[0].message
    print("Assistant (final):")
    print(final_msg.content)

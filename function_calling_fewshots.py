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
def search_database(search_query: str, price_filter: dict | None = None) -> dict[str, str]:
    """Search database for relevant products based on user query"""
    if not search_query:
        raise ValueError("search_query is required")
    if price_filter:
        if "comparison_operator" not in price_filter or "value" not in price_filter:
            raise ValueError("Both comparison_operator and value are required in price_filter")
        if price_filter["comparison_operator"] not in {">", "<", ">=", "<=", "="}:
            raise ValueError("Invalid comparison_operator in price_filter")
        if not isinstance(price_filter["value"], int | float):
            raise ValueError("Value in price_filter must be a number")
    return [{"id": "123", "name": "Example Product", "price": 19.99}]


tool_mapping: dict[str, Callable[..., Any]] = {
    "search_database": search_database,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search database for relevant products based on user query",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "Query string to use for full text search, e.g. 'red shoes'",
                    },
                    "price_filter": {
                        "type": "object",
                        "description": "Filter search results based on price of the product",
                        "properties": {
                            "comparison_operator": {
                                "type": "string",
                                "description": "Operator to compare the column value, either '>', '<', '>=', '<=', '='",  # noqa
                            },
                            "value": {
                                "type": "number",
                                "description": "Value to compare against, e.g. 30",
                            },
                        },
                    },
                },
                "required": ["search_query"],
            },
        },
    }
]

messages: list[dict[str, Any]] = [
    {"role": "system", "content": "You are a product search assistant."},
    {"role": "user", "content": "good options for climbing gear that can be used outside?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {"name": "search_database", "arguments": '{"search_query":"climbing gear outside"}'},
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_abc123",
        "name": "search_database",
        "content": "Search results for climbing gear that can be used outside: ...",
    },
    {"role": "user", "content": "are there any shoes less than $50?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_abc456",
                "type": "function",
                "function": {
                    "name": "search_database",
                    "arguments": '{"search_query":"tenis","price_filter":{"comparison_operator":"<","value":50}}',
                },
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_abc456",
        "name": "search_database",
        "content": "Search results for shoes cheaper than 50: ...",
    },
    {"role": "user", "content": "Find me a red shirt under $20."},
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

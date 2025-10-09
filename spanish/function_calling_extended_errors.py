"""Ejemplo extendido de function calling con manejo robusto de errores.

Este script es similar a `function_calling_extended.py`, pero añade:
 - Parseo seguro de JSON (si el modelo devuelve argumentos malformados)
 - Validación de existencia de la herramienta (mensaje de error si no existe)
 - Captura de excepciones al ejecutar la función
 - Serialización JSON segura del resultado (con fallback)

Muestra un único ciclo (no bucle continuo) pero con protecciones parecidas
al ejemplo de bucle (`function_calling_while_loop.py`).
"""

import json
import os
from collections.abc import Callable
from typing import Any

import azure.identity
import openai
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuración del cliente (Azure, Ollama, GitHub Models o OpenAI.com)
# ---------------------------------------------------------------------------
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
# Implementación de herramientas
# ---------------------------------------------------------------------------
def lookup_weather(city_name: str | None = None, zip_code: str | None = None) -> dict[str, Any]:
    """Busca el clima para una ciudad o código postal (stub determinista)."""
    location = city_name or zip_code or "desconocido"
    return {
        "ubicacion": location,
        "clima": "soleado",
        "temperatura_f": 75,
        "consejo": "¡Gran día para estar al aire libre!",
    }


tool_mapping: dict[str, Callable[..., Any]] = {
    "lookup_weather": lookup_weather,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            # Mantener descripción en inglés para coherencia con otros ejemplos y el schema
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


# ---------------------------------------------------------------------------
# Mensajes iniciales
# ---------------------------------------------------------------------------
messages: list[dict[str, Any]] = [
    {"role": "system", "content": "Eres un chatbot de clima."},
    {"role": "user", "content": "¿Está soleado en Berkeley CA?"},
]

print(f"Modelo: {MODEL_NAME} en Host: {API_HOST}\n")

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages,
    tools=tools,
    tool_choice="auto",
    parallel_tool_calls=False,
)

assistant_msg = response.choices[0].message

if not assistant_msg.tool_calls:
    print("Asistente:")
    print(assistant_msg.content)
else:
    messages.append(
        {
            "role": "assistant",
            "content": assistant_msg.content or "",
            "tool_calls": [tc.model_dump() for tc in assistant_msg.tool_calls],
        }
    )

    for tool_call in assistant_msg.tool_calls:
        fn_name = tool_call.function.name
        raw_args = tool_call.function.arguments or "{}"
        print(f"Solicitud de herramienta: {fn_name}({raw_args})")

        target = tool_mapping.get(fn_name)
        if not target:
            tool_result: Any = f"ERROR: No hay implementación registrada para la herramienta '{fn_name}'"
        else:
            try:
                parsed_args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                parsed_args = {}
                tool_result = "Advertencia: JSON de argumentos malformado; se continúa con argumentos vacíos"
            else:
                try:
                    tool_result = target(**parsed_args)
                except Exception as e:  # noqa: BLE001 - demostración didáctica
                    tool_result = f"Error ejecutando la herramienta {fn_name}: {e}"

        try:
            tool_content = json.dumps(tool_result, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            tool_content = json.dumps({"resultado": str(tool_result)}, ensure_ascii=False)

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": fn_name,
                "content": tool_content,
            }
        )

    followup = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tools,
    )
    final_msg = followup.choices[0].message
    print("Asistente (final):")
    print(final_msg.content)

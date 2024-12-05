import openai
import anthropic
from typing import Optional

def interview(
    api: str,
    system_prompt: str,
    max_tokens: int = 1024,
    temperature: Optional[float] = None,
    api_key: str = None,
    codes: list = ["5j3k", "x7y8"],
    messages: list = None
) -> str:
    # List to store message context
    if messages is None:
        messages = []

    # Prepare function inputs
    api_kwargs = {
        "max_tokens": max_tokens,
    }

    if temperature is not None:
        api_kwargs["temperature"] = temperature

    # Create API specific client, messages, system prompt, and model inputs
    if api == "openai":
        client = openai.OpenAI(api_key=api_key)
        formatted_messages = [{"role": "system", "content": system_prompt}]
        # Add conversation history
        for msg in messages:
            formatted_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        api_kwargs["messages"] = formatted_messages
        api_kwargs["model"] = "gpt-4-0125-preview"
        api_kwargs["stream"] = True
    elif api == "anthropic":
        client = anthropic.Anthropic(api_key=api_key)
        formatted_messages = []
        if messages:
            for msg in messages:
                role = "assistant" if msg["role"] == "assistant" else "user"
                formatted_messages.append({
                    "role": role,
                    "content": msg["content"]
                })
        api_kwargs["messages"] = formatted_messages
        api_kwargs["system"] = system_prompt
        api_kwargs["model"] = "claude-3-sonnet-20240229"
    else:
        raise ValueError("Invalid 'api' value.")

    message_interviewer = ""
    
    if api == "openai":
        response = client.chat.completions.create(**api_kwargs)
        for chunk in response:
            text = chunk.choices[0].delta.content
            if text is not None:
                message_interviewer += text
                yield text
                
    elif api == "anthropic":
        with client.messages.stream(**api_kwargs) as stream:
            for text in stream.text_stream:
                message_interviewer += text
                yield text

    return message_interviewer


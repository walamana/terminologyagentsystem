from typing import Tuple

import backoff
import dotenv
from openai import OpenAI, AsyncOpenAI, RateLimitError
import httpx
import json
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionContentPartParam

dotenv.load_dotenv()

client = AsyncOpenAI()

@backoff.on_exception(backoff.expo, RateLimitError)
async def create_completion_openai(model: str,
                             messages: list[Tuple[str, str]],
                             temperature=0.41,
                             max_completion_tokens=2048,
                             top_p=0.7,
                             frequency_penalty=0,
                             presence_penalty=0,
                             store=False
                             ):
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": role,
                "content": prompt
            } for role, prompt in messages
        ],
        response_format={"type": "text"},
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        store=store
    )
    return response.choices[0].message.content

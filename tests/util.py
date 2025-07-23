import asyncio
from typing import AsyncIterable, Tuple

import dotenv
from openai import OpenAI

SEED = 42

def collect_async(iterable: AsyncIterable):
    """Synchronously collect all items in the AsyncIterable and return them as a list."""
    async def do():
        return [event async for event in iterable]
    return asyncio.run(do())


client: OpenAI | None = None
def get_openai_client() -> OpenAI:
    global client
    if client is None:
        dotenv.load_dotenv()
        client = OpenAI()
    return client


def create_completion_openai_sync(
                             messages: list[Tuple[str, str]],
                             model: str = "gpt-4o-mini",
                             temperature=0.0,
                             max_completion_tokens=2048,
                             top_p=0.0,
                             frequency_penalty=0,
                             presence_penalty=0,
                             store=False,
                            logprobs=False,
                             ):
    response = get_openai_client().chat.completions.create(
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
        store=store,
        logprobs=logprobs,
        seed=SEED,
        top_logprobs=20 if logprobs else None
    )

    if logprobs:
        return response.choices[0].message.content, response.choices[0].logprobs
    else:
        return response.choices[0].message.content

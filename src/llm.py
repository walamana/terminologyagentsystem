from typing import Tuple

import backoff
import dotenv
from openai import AsyncOpenAI, RateLimitError

client: AsyncOpenAI | None = None
def get_openai_client() -> AsyncOpenAI:
    global client
    if client is None:
        dotenv.load_dotenv()
        client = AsyncOpenAI()
    return client

seed = 42

@backoff.on_exception(backoff.expo, RateLimitError)
async def create_completion_openai(
                             messages: list[Tuple[str, str]],
                             model: str = "gpt-4o-mini",
                             temperature=0,
                             max_completion_tokens=2048,
                             top_p=0,
                             frequency_penalty=0,
                             presence_penalty=0,
                             store=False,
                            logprobs=False,
                             ):
    response = await get_openai_client().chat.completions.create(
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
        top_logprobs=5 if logprobs else None,
        seed=seed,
    )

    if logprobs:
        return response.choices[0].message.content, response.choices[0].logprobs
    else:
        return response.choices[0].message.content

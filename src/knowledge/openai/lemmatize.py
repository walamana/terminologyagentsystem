from typing import AsyncIterable

from src.llm import create_completion_openai
from src.prompts.lemmatize import DEVELOPER_PROMPT_SHORT, EXAMPLES
from src.terminology.event import TermExtracted, Event, TermNormalized
from src.terminology.terminology import TermNormalizer


class OpenAILemmatizer(TermNormalizer):

    async def activate(self, event: TermExtracted) -> AsyncIterable[Event]:
        messages = [
            ("system", f"{DEVELOPER_PROMPT_SHORT}"),
            *EXAMPLES,
            # ("user", example_user),
            # ("assistant", output_assistant),
            ("user", event.term.text)
        ]
        response = await create_completion_openai(
            messages=messages,
        )
        event.term.normalization = response
        yield TermNormalized(term=event.term)
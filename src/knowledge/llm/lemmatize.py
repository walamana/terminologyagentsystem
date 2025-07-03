from typing import AsyncIterable

from src.terminology.event import TermExtracted, Event, TermNormalized
from src.terminology.terminology import TermNormalizer

DEVELOPER_PROMPT = """
You are an expert in linguistics and languages.
Your job is to transform words and phrases into a normalized and generalized form.
You transform words and phrases into singular form.
You do not replace words with other similar words.
"""

DEVELOPER_PROMPT_SHORT: str = """
Lemmatize the following term. Keep the word class.
"""

EXAMPLE_USER: list[str] = [
    "örtlicher Zusatz",
    "örtliche Zusätze",
    "Betra",
    "Aufgabe der Triebfahrzeugführerin",
    "Triebfahrzeugführerin",
    "Rangierbegleitender",
]

OUTPUT_ASSISTANT = [
    "örtlicher Zusatz",
    "örtlicher Zusatz",
    "Betra",
    "Aufgabe der Triebfahrzeugführer",
    "Triebfahrzeugführer",
    "Rangierbegleiter",
]

EXAMPLES = [message for input_term, output_term in zip(EXAMPLE_USER, OUTPUT_ASSISTANT) for message in
                [("user", input_term), ("assistant", output_term)]]

class LLMTermLemmatizer(TermNormalizer):

    async def get_llm_response(self, term: str) -> str:
        pass

    async def activate(self, event: TermExtracted) -> AsyncIterable[Event]:
        response = await self.get_llm_response(event.term.text)
        event.term.normalization = response
        yield TermNormalized(term=event.term)
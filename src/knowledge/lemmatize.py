from typing import AsyncIterable

from src.terminology.event import Event, TermExtracted, TermNormalized
from src.llm import create_completion_openai
from src.terminology.terminology import TermNormalizer

developer_prompt = """
You are an expert in linguistics and languages.
Your job is to transform words and phrases into a normalized and generalized form.
You transform words and phrases into singular form.
You do not replace words with other similar words.
"""

developer_prompt_short = """
Lemmatize the following term.
"""


example_user = [
    "örtlicher Zusatz",
    "örtliche Zusätze",
    "Betra",
    "Aufgabe der Triebfahrzeugführerin",
    "Triebfahrzeugführerin",
    "Rangierbegleitender",
    "Aufgabenübertragung an die Rangierbegleiterin"
]

output_assistant = [
    "örtlicher Zusatz",
    "örtlicher Zusatz",
    "Betra",
    "Aufgabe der Triebfahrzeugführer",
    "Triebfahrzeugführer",
    "Rangierbegleiter",
    "Aufgabe"
]

examples = [message for input_term, output_term in zip(example_user, output_assistant) for message in [("user", input_term), ("assistant", output_term)]]

class OpenAILemmatizer(TermNormalizer):

    async def activate(self, event: TermExtracted) -> AsyncIterable[Event]:
        messages = [
            ("system", f"{developer_prompt_short}"),
            *examples,
            # ("user", example_user),
            # ("assistant", output_assistant),
            ("user", event.term.text)
        ]
        response = await create_completion_openai(
            messages=messages,
        )
        event.term.normalization = response
        yield TermNormalized(term=event.term)
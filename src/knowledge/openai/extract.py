from typing import AsyncIterable

from src.llm import create_completion_openai
from src.prompts.extract import DEVELOPER_PROMPT, EXAMPLE_USER, OUTPUT_ASSISTANT
from src.terminology.event import TextExtracted, Event, TermExtracted, OccurrenceResolved
from src.terminology.terminology import TermExtractor


class OpenAIExtractor(TermExtractor):

    async def activate(self, event: TextExtracted) -> AsyncIterable[Event]:
        source = self.blackboard.add_text_source(event.text)
        response = await create_completion_openai(
            messages=[
                ("developer", f"{DEVELOPER_PROMPT}"),
                ("user", EXAMPLE_USER),
                ("assistant", OUTPUT_ASSISTANT),
                ("user", "Input: \n" + event.text)
            ]
        )
        response = response.split("\n")
        terms = [candidate[2:] for candidate in response if candidate.startswith("-") or candidate.startswith("*")]



        for term in terms:
            t = self.blackboard.find_term(term_str=term)
            if t is None:
                t = self.blackboard.add_term(term=term)
            t.occurrences.append(source.id)
            yield TermExtracted(term=t)
            yield OccurrenceResolved(term=t, source=source)

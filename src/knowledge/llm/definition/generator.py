import asyncio
import re
from typing import AsyncIterable, Annotated

from pydantic import Field

from src.logger import simple_custom_logger
from src.terminology.event import Event
from src.terminology.models import Term
from src.terminology.terminology import Definition, DefinitionGenerator, PartialDefinitionGenerated, OccurrenceResolved

DEVELOPER_PROMPT = """
Erstelle eine Definition für einen Begriff anhand von gegebenen Textausschnitten.
Bleibe präzise und kurz. Nutze nur die Informationen aus dem gegebenen Text. Nutze kein gelerntes Wissen aus deinen Trainingsdaten!
Wenn nicht genug Information vorhanden ist oder die Definition zu generell, vage oder nicht fachspezifisch ist, gebe "ERROR" aus.
"""

logger = simple_custom_logger("DEFGEN")

class LLMDefinitionGenerator(DefinitionGenerator):

    WINDOW_START: int = 200
    WINDOW_END: int = 300
    MIN_OVERLAP: int = 100
    MAX_LENGTH: int = 1000

    CERTAINTY_THRESHOLD: int = 0.05

    known_sources: Annotated[dict[str, list[Term]], Field(default_factory=dict[str, list[Term]])]

    async def generate_definition_from_source(self, term: str, context: str) -> str | None:
        pass

    def get_matches(self, term: str, text: str):
        pattern = rf"{term}"
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if len(matches) == 0:
            return [text]

        excerpts = []
        last_start = 0
        last_end = 0
        for match in matches:
            start = max(0, match.start() - self.WINDOW_START)
            end = min(len(text), match.end() + self.WINDOW_END)

            overlap = last_end - start
            length = end - last_start
            if overlap > self.MIN_OVERLAP and length <= self.MAX_LENGTH:
                if len(excerpts) == 0:
                    excerpts.append(text[start:end])
                else:
                    excerpts[-1] = text[last_start:end]
                last_end = end
            else:
                last_start = start
                last_end = end
                excerpts.append(text[start:end])
        return excerpts



    async def activate(self, event: OccurrenceResolved) -> AsyncIterable[Event]:
        if str(event.source.id) not in self.known_sources:
            self.known_sources[str(event.source.id)] = list()

        if event.term in self.known_sources[str(event.source.id)]:
            return

        self.known_sources[str(event.source.id)].append(event.term)

        tasks = []
        async with asyncio.TaskGroup() as tg:
            matches = self.get_matches(term=event.term.normalized_or_text(), text=event.source.text)
            if len(matches) > 0:
                for match in matches:
                    task = tg.create_task(self.generate_definition_from_source(event.term.normalized_or_text(), match))
                    tasks.append(task)
            else:
                tg.create_task(self.generate_definition_from_source(event.term.normalized_or_text(), event.source.text))


        for task in asyncio.as_completed(tasks):
            result = await task
            # print(f"Resolved for {event.term.normalized_or_text()}: {result}")
            if result is not None:
                definition = Definition(
                    text=result,
                    verified=False,
                    partial=True,
                    source=event.source
                )
                event.term.definitions.append(definition)
                yield PartialDefinitionGenerated(
                    term=event.term,
                    definition=definition
                )

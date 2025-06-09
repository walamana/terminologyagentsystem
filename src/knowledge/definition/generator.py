import asyncio
from typing import AsyncIterable

from src.service.event import Event
from src.llm import create_completion_openai
from src.service.terminology import Definition, DefinitionGenerator, PartialDefinitionGenerated, OccurrenceResolved
import re

developer_prompt = """
Erstelle eine Definition für einen Begriff anhand von gegebenen Textausschnitten.
Bleibe präzise und kurz. Nutze nur die Informationen aus dem gegebenen Kontext. 
Wenn nicht genug Information vorhanden ist oder zu generell, vage oder nicht fachspezifisch ist, gebe "ERROR" aus.
"""

class OpenAIDefinitionGenerator(DefinitionGenerator):

    WINDOW_START: int = 200
    WINDOW_END: int = 300

    async def generate_definition_from_source(self, term: str, context: str) -> str | None:
        response = await create_completion_openai(
            model="gpt-4o-mini",
            messages=[
                ("system", f"{developer_prompt}"),
                ("user", f"{context}"),
                ("user", f"Definiere den Begriff \"{term}\"."),
            ],
        )

        if response == "ERROR":
            return None
        return response


    async def activate(self, event: OccurrenceResolved) -> AsyncIterable[Event]:
        # print(f"Find {event.term.normalized_or_text()} in \"{event.source.text}...\"")

        pattern = rf"{event.term.normalized_or_text()}"
        # print(f"Pattern: {pattern}")
        # print(f"In {event.term.normalized_or_text() in event.source.text}")

        tasks = []
        async with asyncio.TaskGroup() as tg:
            for match in re.finditer(pattern, event.source.text, re.IGNORECASE):
                context = event.source.text[max(0, match.start() - self.WINDOW_START):min(match.end() + self.WINDOW_END, len(event.source.text))]
                task = tg.create_task(self.generate_definition_from_source(event.term.normalized_or_text(), context))
                tasks.append(task)

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


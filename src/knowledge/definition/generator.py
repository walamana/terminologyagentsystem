import asyncio
from math import exp
from typing import AsyncIterable

import numpy as np

from src.logger import simple_custom_logger
from src.terminology.event import Event
from src.llm import create_completion_openai
from src.terminology.models import Term
from src.terminology.terminology import Definition, DefinitionGenerator, PartialDefinitionGenerated, OccurrenceResolved, \
    Blackboard
import re

developer_prompt = """
Erstelle eine Definition für einen Begriff anhand von gegebenen Textausschnitten.
Bleibe präzise und kurz. Nutze nur die Informationen aus dem gegebenen Kontext. 
Wenn nicht genug Information vorhanden ist oder zu generell, vage oder nicht fachspezifisch ist, gebe "ERROR" aus.
"""

logger = simple_custom_logger("DEFGEN")

class OpenAIDefinitionGenerator(DefinitionGenerator):

    WINDOW_START: int = 200
    WINDOW_END: int = 300

    CERTAINTY_THRESHOLD: int = 0.05

    async def generate_definition_from_source(self, term: str, context: str) -> str | None:
        response, log_probs = await create_completion_openai(
            messages=[
                ("developer", f"{developer_prompt}"),
                ("user", f"{context}"),
                ("user", f"Definiere den Begriff \"{term}\"."),
            ],
            logprobs=True
        )

        for token in log_probs.content[0].top_logprobs:
            prob = np.exp(token.logprob)
            if token.token == "ERROR" and prob > self.CERTAINTY_THRESHOLD:
                # logger.debug(f"Generation uncertain. Probability of 'ERROR' token {prob}>{self.CERTAINTY_THRESHOLD}!")
                return None

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



if __name__ == "__main__":
    blackboard = Blackboard()
    generator = OpenAIDefinitionGenerator(
        blackboard=blackboard
    )

    context = """

Abstellen

Züge und Triebfahrzeuge sind abgestellt, wenn sie nicht mit einem Triebfahrzeugführer besetzt sind oder nicht gesteuert werden. Wagen sind abgestellt, sofern sie nicht in Züge eingestellt sind oder nicht rangiert werden.

Abstoßen

Abstoßen ist das Bewegen geschobener, nicht mit einem arbeitenden Triebfahrzeug gekuppelter Fahrzeuge durch Beschleunigen, so dass die Fahrzeuge allein weiterfahren, nachdem das Triebfahrzeug angehalten hat.

""".strip()

    term = blackboard.add_term("Abstellen")
    source = blackboard.add_text_source(text=context)

    async def test():
        async for event in generator.activate(OccurrenceResolved(term=term, source=source)):
            print(f"Event {event}")

    asyncio.run(test())
import asyncio
import re
from typing import Annotated, AsyncIterable
from uuid import UUID

from pydantic import Field

from src.llm import create_completion_openai
from src.logger import simple_custom_logger
from src.terminology.event import OccurrenceResolved, Event
from src.terminology.terminology import DefinitionGenerator, Blackboard

logger = simple_custom_logger("UNIGEN")

prompt_introduction = """
Erstelle eine Definition für den Begriff "%term%" anhand von gegebenen Textausschnitten.
Bleibe präzise und kurz. Nutze nur die Informationen aus dem gegebenen Kontext. 
Wenn nicht genug Information vorhanden ist oder zu generell, vage oder nicht fachspezifisch ist, gebe "ERROR" aus.
Füge in die Definition die jeweiligen Referenzen hinzu, indem du die Nummer des Abschnitts verwendest im Format [<nummer>].
""".strip()

class OpenAIUnifiedDefinitionGenerator(DefinitionGenerator):

    MIN_OCCURRENCES: int = 3

    WINDOW_START: int = 100
    WINDOW_END: int = 200

    locks: Annotated[dict[UUID, asyncio.Lock], Field(default_factory=lambda: {})]

    # TODO: See Prompt Engineering for LLMs -> Elastic Snippets
    async def activate(self, event: OccurrenceResolved) -> AsyncIterable[Event]:
        if event.term.id not in self.locks:
            self.locks[event.term.id] = asyncio.Lock()

        async with self.locks[event.term.id]:
            logger.info(f"Locking {event.term.normalized_or_text()} for unified definition generator")

            term = event.term.normalized_or_text()
            pattern = rf"{term}"
            context = []
            for source_id in event.term.occurrences:
                source = self.blackboard.get_text_source(id=source_id)
                # FIXME: Create elastic snippet? -> dynamic window length? -> differences in quality?

                # Find all occurrences of the term
                snippets = []

                current_start = 0
                current_end = 0
                snippet = ""

                matches = list(re.finditer(pattern, source.text, re.IGNORECASE))

                for match in matches:
                    start = max(0, match.start() - self.WINDOW_START)
                    end = match.end() + self.WINDOW_END
                    if start < current_end:
                        # snippet overlaps with current
                        current_end = end
                        pass
                    else:
                        # snippet is further away -> new one
                        if snippet != "":
                            snippets.append(snippet)
                        current_start = start
                        current_end = end
                    snippet = source.text[current_start:current_end]
                if snippet != "":
                    snippets.append(snippet)

                context += snippets

            # for snippet in context:
            #     logger.debug(f"Snippet: {snippet}")

            logger.debug(f"Found {len(snippets)} snippets for {term}")

            # TODO: think about: how can cost be reduced? How can I decide if a text is relevant to a term?

            messages = [
                ("system", f"{prompt_introduction.replace('%term%', term)}"),
                ("user", f"Hier sind einige Textausschnitte, die du verwenden kannst. Beziehe dich bei der Generation nur auf Wissen aus den Textstellen!"),
                ("user", "\n\n".join([f"[{index}] {context}" for index, context in enumerate(context)])),
                ("user", f"Definiere den Begriff \"{term}\". Beziehe dich nur auf die Textabschnitte!"),
            ]

            # logger.debug("\n----\n".join([text for _, text in messages]))

            result = await create_completion_openai(
                messages=messages,
                model="o4-mini"
            )

            print(result)





        yield


if __name__ == "__main__":
    blackboard = Blackboard(
        terms=[],
        sources=[]
    )

    term = blackboard.add_term("Fahrdienstleiter")

    with open("./../../../data/Handbuch-40820-data_11-15-1.txt", "r") as f:
        source1 = blackboard.add_text_source(text=f.read())

    with open("./../../../data/Handbuch-40820-data_11-15-5.txt", "r") as f:
        source2 = blackboard.add_text_source(text=f.read())

    with open("./../../../data/Handbuch-40820-data.txt", "r") as f:
        source3 = blackboard.add_text_source(text=f.read())

    term.occurrences = [source.id for source in [source1, source2, source3]]

    generator = OpenAIUnifiedDefinitionGenerator(blackboard=blackboard)

    async def test():
        async for event in generator.activate(OccurrenceResolved(term=term, source=source3)):
            pass

    asyncio.run(test())

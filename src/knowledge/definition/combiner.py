import asyncio
from typing import Annotated, AsyncIterable
from uuid import UUID

from pydantic import Field

from src.knowledge.definition.generator import PartialDefinitionGenerated
from src.terminology.event import Event, CombinedDefinitionGenerated
from src.llm import create_completion_openai
from src.logger import simple_custom_logger
from src.terminology.models import Definition, Term
from src.terminology.terminology import DefinitionCombiner, Blackboard

logger = simple_custom_logger("COMBINER")

class OpenAIDefinitionCombiner(DefinitionCombiner):

    MIN_PARTIAL_DEFINITIONS: int = 3

    locks: Annotated[dict[UUID, asyncio.Lock], Field(default_factory=lambda: {})]
    lock: asyncio.Lock = asyncio.Lock()

    async def activate(self, event: PartialDefinitionGenerated) -> AsyncIterable[Event]:
        # Since definitions for a term can be generated concurrently, definitions might get combined multiple times
        # For now, only one definition can be combined at once.
        # FIXME: Improve locking (lock per term?)
        if event.term.id not in self.locks:
            self.locks[event.term.id] = asyncio.Lock()
        async with self.locks[event.term.id]:
            logger.info(f"Locking {event.term.normalized_or_text()} definition combiner")
            has_verified_definition = next((definition for definition in event.term.definitions if definition.verified), None) is not None

            partial_definitions = [definition for definition in event.term.definitions if definition.is_partial()]

            if not has_verified_definition and len(partial_definitions) >= self.MIN_PARTIAL_DEFINITIONS:

                event.term.definitions = [definition for definition in event.term.definitions if not definition.is_combined()]

                async with asyncio.TaskGroup() as tg:
                    tasks = []
                    for definition in event.term.definitions:
                        task = tg.create_task(create_completion_openai(
                            messages=[
                                ("user", f"""Ist der folgende Text eine Definition für den Begriff \"{event.term.normalized_or_text()}\"? Wenn die Definition spezifisch genug ist, beende deine Folgerung mit TRUE, ansonsten mit FALSE. 
                                
                                {definition.text}"""),
                            ],
                        ))
                        tasks.append((definition, task))

                relevant_definitions = [definition for definition, task in tasks if task.result().endswith("TRUE")]
                logger.debug(f"Relevant definitions: {relevant_definitions}")
                relevant_definitions_text = "\n\n".join([definition.text for definition in relevant_definitions])
                response = await create_completion_openai(
                    messages=[
                        ("system", """Nutze nur das gegebene Wissen aus den Anfragen."""),
                        ("user", f"""Erstelle eine kombinierte Definition für \"{event.term.normalized_or_text()}\" anhand der folgenden Definitionen. 
                        Starte mit allgemeinen Informationen und werde dann spezifischer. Verwende nur die Informationen aus den unten stehenden Texten.
    
                        {relevant_definitions_text}"""),
                    ],
                )

                combined_definition = Definition(
                    text=response,
                    verified=False,
                    partial=False,
                    source=relevant_definitions
                )

                event.term.definitions.append(combined_definition)

                yield CombinedDefinitionGenerated(
                    term=event.term,
                    combined_definition=combined_definition,
                    relevant_definitions=relevant_definitions
                )
            logger.info(f"Lock released for {event.term.normalized_or_text()}")



if __name__ == '__main__':
    blackboard = Blackboard()
    combiner = OpenAIDefinitionCombiner(blackboard=blackboard)

    term = Term(
        text="Sperrfahrt",
        normalization="Sperrfahrt",
        occurrences=[],
        definitions=[
            Definition(
                text="Eine Sperrfahrt ist ein Zug, der anstelle des Begriffs 'Zug' bezeichnet wird, wenn es sich um eine spezielle Fahrt handelt, die nicht regulär verkehrt.",
                partial=True,
                verified=False,
                source=None
            ),
            Definition(
                text="Eine Sperrfahrt ist ein Zug, der in Aufträgen oder Meldungen anstelle des Begriffs 'Zug' verwendet wird, wenn es sich um eine spezielle Fahrt handelt, die nicht regulär im Fahrplan enthalten ist.",
                partial=True,
                verified=False,
            ),
            Definition(
                text="Eine Sperrfahrt ist eine spezielle Art von Zugfahrt, die anstelle des Begriffs \"Zug\" verwendet wird, wenn es sich um eine Sperrfahrt handelt.",
                partial=True,
                verified=False,
            )
        ]
    )

    async def test():
        async for event in combiner.activate(PartialDefinitionGenerated(term=term, definition=term.definitions[2])):
            print(f"Event: {event}")
            print(f"Definition: {event.combined_definition}")

    asyncio.run(test())


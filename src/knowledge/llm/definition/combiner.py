import asyncio
from typing import Annotated, AsyncIterable
from uuid import UUID

from pydantic import Field

from src.logger import simple_custom_logger
from src.terminology.event import Event, CombinedDefinitionGenerated, PartialDefinitionGenerated
from src.terminology.models import Definition
from src.terminology.terminology import DefinitionCombiner

logger = simple_custom_logger("COMBINER")

RELEVANCE_USER_PROMPT = """Ist der folgende Text eine Definition für den Begriff \"%term%\"? Wenn die Definition spezifisch genug ist, beende deine Folgerung mit TRUE, ansonsten mit FALSE.

%definition%"""

COMBINE_SYSTEM_PROMPT = """Nutze nur das gegebene Wissen aus den Anfragen."""
COMBINE_USER_PROMPT="""Erstelle eine kombinierte Definition für \"%term%\" anhand der folgenden Definitionen. Starte mit allgemeinen Informationen und werde dann spezifischer. Verwende nur die Informationen aus den unten stehenden Texten.

%definitions%"""

class LLMDefinitionCombiner(DefinitionCombiner):

    MIN_PARTIAL_DEFINITIONS: int = 3

    locks: Annotated[dict[UUID, asyncio.Lock], Field(default_factory=lambda: {})]
    lock: asyncio.Lock = asyncio.Lock()

    async def get_llm_response_relevance(self, term: str, definition: str) -> str:
        pass

    async def get_llm_response_combine(self, term: str, definitions: str) -> str:
        pass

    async def activate(self, event: PartialDefinitionGenerated) -> AsyncIterable[Event]:
        # Since definitions for a term can be generated concurrently, definitions might get combined multiple times
        # For now, only one definition can be combined at once.
        # FIXME: Improve locking (lock per term?)
        if event.term.id not in self.locks:
            self.locks[event.term.id] = asyncio.Lock()
        async with self.locks[event.term.id]:
            logger.info(f"Locking {event.term.normalized_or_text()} definition combiner {id(event)}")
            has_verified_definition = next((definition for definition in event.term.definitions if definition.verified), None) is not None

            partial_definitions = [definition for definition in event.term.definitions if definition.is_partial()]

            if not has_verified_definition and len(partial_definitions) >= self.MIN_PARTIAL_DEFINITIONS:

                event.term.definitions = [definition for definition in event.term.definitions if not definition.is_combined()]

                async with asyncio.TaskGroup() as tg:
                    tasks = []
                    for definition in event.term.definitions:
                        task = tg.create_task(self.get_llm_response_relevance(event.term.normalized_or_text(), definition.text))
                        tasks.append((definition, task))

                relevant_definitions = [definition for definition, task in tasks if task.result().endswith("TRUE")]
                logger.debug(f"Relevant definitions: {relevant_definitions}")
                relevant_definitions_text = "\n\n".join([definition.text for definition in relevant_definitions])
                response = await self.get_llm_response_combine(event.term.normalized_or_text(), relevant_definitions_text)

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



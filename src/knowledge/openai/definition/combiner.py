import asyncio
from typing import Annotated
from uuid import UUID

from pydantic import Field

from src.knowledge.llm.definition.combiner import LLMDefinitionCombiner, RELEVANCE_USER_PROMPT, COMBINE_SYSTEM_PROMPT, \
    COMBINE_USER_PROMPT
from src.llm import create_completion_openai
from src.terminology.event import PartialDefinitionGenerated
from src.terminology.models import Definition, Term
from src.terminology.terminology import Blackboard


class OpenAIDefinitionCombiner(LLMDefinitionCombiner):

    MIN_PARTIAL_DEFINITIONS: int = 3

    locks: Annotated[dict[UUID, asyncio.Lock], Field(default_factory=lambda: {})]
    lock: asyncio.Lock = asyncio.Lock()

    async def get_llm_response_relevance(self, term: str, definition: str) -> str:
        return await create_completion_openai(
            messages=[
                ("user", RELEVANCE_USER_PROMPT.replace("%term%", term).replace("%definition%", definition)),
            ],
        )

    async def get_llm_response_combine(self, term: str, definitions: str) -> str:
        return await create_completion_openai(
            messages=[
                ("system", COMBINE_SYSTEM_PROMPT),
                ("user", COMBINE_USER_PROMPT.replace("%term%", term).replace("%definitions%", definitions)),
            ],
        )





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


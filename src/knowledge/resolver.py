from typing import Annotated, AsyncIterable, Optional, Any

from pydantic import Field

from src.terminology.event import Event, VerifiedDefinitionResolved
from src.terminology.models import TextSource, Definition
from src.terminology.terminology import DefinitionResolver


class CSVDefinitionResolver(DefinitionResolver):
    definitions: Annotated[dict, Field(default_factory=dict)]
    source: Optional[TextSource] = None

    def model_post_init(self, __context: Any) -> None:
        langs = ["de"]
        for lang in langs:
            with open(f"data/{lang}-glossary.csv", "r") as f:
                data = f.read().split("\n")
                for row in data:
                    key, value = row.split("\t")
                    self.definitions[key] = value
        self.source = self.blackboard.add_text_source("DICTIONARY")

    async def activate(self, event: Event) -> AsyncIterable[Event]:
        term = event.term
        term_str = term.normalized_or_text()
        if term_str in self.definitions:
            definition = Definition(
                text = self.definitions[term_str],
                verified=True,
                partial=False,
                source=self.source,
            )
            term.definitions.append(definition)
            yield VerifiedDefinitionResolved(term=term, definition=definition)

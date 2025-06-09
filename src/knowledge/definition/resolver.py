import csv

from src.service.event import Event
from src.service.terminology import Definition, TextSource, \
    DefinitionResolver, VerifiedDefinitionResolved

from typing import Annotated, AsyncIterable, Optional, Any

from pydantic import Field

class CSVDefinitionResolver(DefinitionResolver):
    definitions: Annotated[dict, Field(default_factory=dict)]
    source: Optional[TextSource] = None

    def model_post_init(self, __context: Any) -> None:
        langs = ["de", "en"]
        for lang in langs:
            with open(f"data/{lang}-glossary.csv", "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.definitions[row["0"]] = row["1"]
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

from typing import Annotated, List

from pydantic import Field

from src.terminology.terminology import TermExtracted, OccurrenceResolver, OccurrenceResolved



class MockOccurrenceResolver(OccurrenceResolver):

    texts: Annotated[List[str], Field(default_factory=lambda:[
        "Das ist ein Text über den Schrankenwärter",
        "Das Gleis ist noch nicht sichern",
        "Wir fahren hier eine Sperrfahrt."
        "Die Strecke muss man sichern."
    ])]

    async def activate(self, event: TermExtracted):
        term = event.term
        for text in self.texts:
            if term.text in text:
                source = self.blackboard.add_text_source(text)
                term.occurrences.append(source.id)
                yield OccurrenceResolved(term=term, source=source)

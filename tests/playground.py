import asyncio
import time
from typing import Annotated, List, Type, AsyncIterable, Dict

from pydantic import Field

from src.knowledge.definition.generator import OpenAIDefinitionGenerator

from src.knowledge.document import Pdf2Text
from src.terminology.event import Event
from src.knowledge.extract import OpenAIExtractor
from src.terminology.terminology import Controller, KnowledgeSource, \
    TermExtracted, Term, TextExtracted


class TermDefined(Event):
    term: Term
    definition: str

class DBDictionary(KnowledgeSource):
    handles: Annotated[List[Type[Event]], Field(default_factory=lambda:[TermExtracted])]
    terms: Annotated[Dict[str, str], Field(default_factory=lambda:{
        "Sperrfahrt": "Fahrt auf einem eigentlich gesperrten Gleisbereich.",
        "Gleis": "Darauf fahren Züge."
    })]


    async def activate(self, event: TermExtracted) -> AsyncIterable[Event]:
        await asyncio.sleep(2)
        if event.term.text in self.terms:
            print(f"Term {event.term.text} defined {self.terms[event.term.text]}")
            yield TermDefined(term=event.term, definition=self.terms[event.term.text])


class TextExporter(KnowledgeSource):
    handles: Annotated[List[Type[Event]], Field(default_factory=lambda:[TextExtracted])]

    counter: int = 0

    async def activate(self, event: TextExtracted) -> AsyncIterable[Event]:
        path = f"export/text_{self.counter}.md"
        self.counter += 1
        with open(path, "w") as f:
            f.write(event.text)
        yield

if __name__ == '__main__':

    controller = Controller()

    controller.register_knowledge_source(Pdf2Text)
    controller.register_knowledge_source(TextExporter)
    # controller.register_knowledge_source(CValue)
    controller.register_knowledge_source(OpenAIExtractor)
    # controller.register_knowledge_source(OpenAILemmatizer)
    # controller.register_knowledge_source(OccurrenceResolver)
    # controller.register_knowledge_source(DBDictionary)
    controller.register_knowledge_source(OpenAIDefinitionGenerator)

    # asyncio.run(controller.start())
    with open("../export/text_0.md", "r") as f:
        text = f.read()

    test_prompt_rothmann = """Brauch i mal 3 Führer, da erste nimmd si 247 ZickZack 211/2 Kuppeln Voll 39798, die anderen zwei nehmen sich den Vollzug im Waschgleis in den Westen zwei Teile, der erste geht nach 205/1 kuppeln Lang 39702. Der zweite vom Westen geht Osten Halle 5/3"""

    print("Started TAS")
    start = time.time()

    # asyncio.run(controller.emit(TextExtracted(text=test_prompt_rothmann)))

    # term = Term(text="Vollzug")
    # asyncio.run(controller.emit(OccurrenceResolved(term=term, source=TextSource(text=test_prompt_rothmann))))

    asyncio.run(controller.analyse_document(path="../data/Handbuch-40820-data_37.pdf"))

    end = time.time()
    print(f"TAS finished in {end - start} seconds.")

    for term in sorted(controller.blackboard.terms, key=lambda t: t.text):
        print(f"{term.normalization} ({term.text})")
        for definition in term.definitions:
            print(f"{'~' if definition.partial else '-'} {definition.text}")
        for occ in term.occurrences:
            preview = occ[:100].replace("\n", "\\n")
            print(f"> {preview}")

    # print(controller.blackboard)

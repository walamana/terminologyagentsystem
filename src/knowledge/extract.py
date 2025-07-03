import asyncio
from typing import AsyncIterable, Any

import spacy
from spacy import Language

from src.logger import simple_custom_logger
from src.terminology.event import Event, TextExtracted, TermExtracted
from src.terminology.terminology import TermExtractor, OccurrenceResolved, Blackboard
from src.utils import lazy_module

logger = simple_custom_logger("TERMEXTRACTOR")

class CValue(TermExtractor):

    nlp: Language = None

    def model_post_init(self, __context: Any) -> None:
        self.nlp = spacy.load("de_core_news_md")
        lazy_module("pyate").TermExtraction.configure({
            "language": "de",
            "model_name": "de_core_news_md",
            "MAX_WORD_LENGTH": 5
        })

    async def activate(self, event: TextExtracted) -> AsyncIterable[Event]:
        candidates = lazy_module("pyate").cvalues(event.text, have_single_word=True).to_dict().keys()
        source = self.blackboard.add_text_source(event.text)
        for term in candidates:
            t = self.blackboard.add_term(term)
            yield OccurrenceResolved(term=t, source=source)
            yield TermExtracted(term=t)


if __name__ == "__main__":
    blackboard = Blackboard()
    extractor = CValue(blackboard=blackboard)

    text = "Wenn im Zug außergewöhnliche Sendungen oder außergewöhnliche Fahrzeuge eingestellt sind, müssen sich deren Beförderungsanordnungen beim Zug befinden und die Nummern der Beförderungsanordnungen dem Fahrdienstleiter mitgeteilt worden sein."

    async def run():
        async for event in extractor.activate(TextExtracted(text=text)):
            print(event.term.normalized_or_text())

    asyncio.run(run())
from typing import AsyncIterable, Any

import pyate
import spacy
from spacy import Language

from src.logger import simple_custom_logger
from src.terminology.event import Event, TextExtracted, TermExtracted
from src.terminology.terminology import TermExtractor, OccurrenceResolved

logger = simple_custom_logger("TERMEXTRACTOR")

class CValue(TermExtractor):

    nlp: Language

    def model_post_init(self, __context: Any) -> None:
        self.nlp = spacy.load("de_core_news_md")
        pyate.TermExtraction.configure({
            "language": "de",
            "model_name": "de_core_news_md",
            "MAX_WORD_LENGTH": 4
        })

    async def activate(self, event: TextExtracted) -> AsyncIterable[Event]:
        candidates = pyate.cvalues(event.text, have_single_word=True).to_dict().keys()
        source = self.blackboard.add_text_source(event.text)
        for term in candidates:
            t = self.blackboard.add_term(term)
            yield OccurrenceResolved(term=t, source=source)
            yield TermExtracted(term=t)



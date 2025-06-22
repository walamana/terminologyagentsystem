from typing import AsyncIterable, Any

import numpy as np
import pyate
import spacy
from spacy import Language

from src.llm import create_completion_openai
from src.logger import simple_custom_logger
from src.terminology.event import Event, TextExtracted, TermExtracted

from src.terminology.terminology import TermExtractor, OccurrenceResolved

logger = simple_custom_logger("TERMEXTRACTOR")

developer_prompt = """
Du bist Experte für Terminologie der Eisenbahnen in Europa, insbesondere in Deutschland. 
Deine Aufgabe besteht darin, aus einem Text Begriffe, Abkürzungen und Phrasen zu extrahieren. 
Du extrahierst nur Terminologie, die wahrscheinlich in der Eisenbahn verwendet wird.
Du erkennst Abkürzungen und behällst sie unverändert bei.
Du verwendest die Lemma der jeweiligen Wörter. Du wandelst Wörter in Singular um.
Du extrahierst Phrasen und Wörter sowie verschachtelte Begriffe und deren Einzelteile.
Achte bei längeren Phrasen darauf, ob aus dem Text klar wird, dass es sich um einen besonderen Begriff handelt, der Wahrscheinlich verwendet wird.
Beginne mit den Begriffen, die am wahrscheinlichsten relevant sind.
Gib nur eine Liste von Begriffen zurück. Extrahiere nur Begriffe, die besonders für den Kontext "Bahn" sind!
"""

example_user = """
Input:
Du musst das Hauptsignal auf Fahrt stellen.
"""

output_assistant = """
Output:
- Hauptsignal auf Fahrt stellen
- Hauptsignal
- auf Fahrt stellen
- Fahrtstellung eines Hauptsignals
"""



class OpenAIExtractor(TermExtractor):

    async def activate(self, event: TextExtracted) -> AsyncIterable[Event]:
        source = self.blackboard.add_text_source(event.text)
        response = await create_completion_openai(
            messages=[
                ("developer", f"{developer_prompt}"),
                ("user", example_user),
                ("assistant", output_assistant),
                ("user", "Input: \n" + event.text)
            ]
        )
        response = response.split("\n")
        terms = [candidate[2:] for candidate in response if candidate.startswith("-") or candidate.startswith("*")]



        for term in terms:
            t = self.blackboard.find_term(term_str=term)
            if t is None:
                t = self.blackboard.add_term(term=term)
            t.occurrences.append(source.id)
            yield TermExtracted(term=t)
            yield OccurrenceResolved(term=t, source=source)



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
        for term in candidates:
            t = self.blackboard.add_term(term)
            yield TermExtracted(term=t)



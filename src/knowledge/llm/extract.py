import re
from typing import AsyncIterable

from src.terminology.event import TextExtracted, Event, TermExtracted, OccurrenceResolved
from src.terminology.terminology import TermExtractor

DEVELOPER_PROMPT: str = """
Du bist Experte für Terminologie und Fachbegriffe. 
Deine Aufgabe besteht darin, aus einem Text Begriffe, Abkürzungen und Phrasen zu extrahieren. 
Du extrahierst nur Terminologie, die wahrscheinlich in der Eisenbahn verwendet wird.
Du erkennst Abkürzungen und behällst sie unverändert bei. Nur wenn die vollständige Form vorhanden ist, fügst du sie in Klammern am Ende des Begriffs an.
Du extrahierst Phrasen und Wörter sowie verschachtelte Begriffe und deren Einzelteile.
Achte bei längeren Phrasen darauf, ob aus dem Text klar wird, dass es sich um einen besonderen Begriff handelt, der Wahrscheinlich verwendet wird.
Beginne mit den Begriffen, die am wahrscheinlichsten relevant sind.
Gib nur eine Liste von Begriffen zurück. Extrahiere nur Begriffe, die besonders für den Kontext "Eisenbahn" sind!
"""

EXAMPLE_USER: str = """
Input:
Du musst das Hauptsignal auf Fahrt stellen.
"""

OUTPUT_ASSISTANT: str = """
Output:
- Hauptsignal auf Fahrt stellen
- Hauptsignal
- auf Fahrt stellen
- Fahrtstellung eines Hauptsignals
"""

class LLMTermExtractor(TermExtractor):


    async def get_llm_response(self, text: str) -> str:
        pass

    async def activate(self, event: TextExtracted) -> AsyncIterable[Event]:
        source = self.blackboard.add_text_source(event.text)
        response = await self.get_llm_response(event.text)
        response = response.split("\n")
        terms = [candidate[2:] for candidate in response if candidate.startswith("-") or candidate.startswith("*")]

        for term in terms:

            variation_match = re.search(r"\(.+\)$", term)
            abbreviation = None

            if variation_match:
                variation = (variation_match.group(0)
                                .replace("(", "")
                                .replace(")", "")
                                .strip())
                term = term.replace(variation_match.group(0), "").strip()
                if len(variation) > len(term):
                    abbreviation = term
                    term = variation
                else:
                    abbreviation = variation
                    term = term

            t = self.blackboard.find_term(term_str=term)

            if t is None:
                t = self.blackboard.add_term(term=term)

            if abbreviation and not abbreviation in t.variations:
                t.variations.append(abbreviation)

            t.occurrences.append(source.id)
            yield TermExtracted(term=t)
            yield OccurrenceResolved(term=t, source=source)

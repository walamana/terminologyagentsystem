from typing import AsyncIterable, Any

import pyate
import spacy
from spacy import Language

from src.llm import create_completion_openai
from src.service.event import Event

from src.service.terminology import TextExtracted, TermExtracted, TermExtractor, OccurrenceResolved

developer_prompt = """
You are an expert in terminology of the rail service agencies in Europe, especially Germany. 
Your job is to extract term candidates, abbreviations and phrases from a text. 
You focus on terminology that is used in rail service in Germany.
You create different grammatical versions of a term or phrase if applicable. 
You recognize abbreviations and keep them as they are.
You use the lemma of the words in use. You transform them into singular form.
You extract phrases and words as well as nested terms and its single parts.
Only return a list of terms.
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
            model="gpt-4o-mini",
            messages=[
                ("system", f"{developer_prompt}\n{example_user}\n{output_assistant}"),
                # ("user", example_user),
                # ("assistant", output_assistant),
                ("user", "Input: \n" + event.text)
            ],
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



from unittest import TestCase

from src.knowledge.openai.extract import OpenAIExtractor
from src.terminology.event import TextExtracted, TermExtracted, OccurrenceResolved
from src.terminology.terminology import Blackboard
from tests.util import collect_async


class TestTermExtractor(TestCase):

    def test_common(self):
        blackboard = Blackboard(terms=[], sources=[])
        extractor = OpenAIExtractor(blackboard=blackboard)

        input = """Einseitig gerichtete Sprecheinrichtung verwenden\n Aufträge dürfen über einseitig gerichtete Sprecheinrichtungen gegeben werden, wenn dies im Einzelfall nicht verboten ist und der Empfänger die Ausführung melden muss oder der Auftraggeber die Ausführung selbst erkennen kann. Meldungen dürfen über einseitig gerichtete Sprecheinrichtungen nicht gegeben werden."""

        expected = ["Einseitig gerichtete Sprecheinrichtung", "Aufträge", "Empfänger", "Auftraggeber", "Meldungen"]

        initial_event = TextExtracted(text=input)
        actual_events = collect_async(extractor.activate(initial_event))

        actual_events_extracted = [event for event in actual_events if type(event) is TermExtracted]
        actual_events_occurrence = [event for event in actual_events if type(event) is OccurrenceResolved]

        actual_terms_text = set(event.term.text.lower() for event in actual_events_extracted)

        missing_terms = []
        for term in expected:
            if term.lower() not in actual_terms_text:
                missing_terms.append(term)

        if len(missing_terms) > 0:
            self.fail(f"Missing terms [{', '.join(missing_terms)}] in extracted events ([{', '.join(actual_terms_text)}]).")

        for term in actual_terms_text:
            if len([event for event in actual_events_extracted if event.term.text.lower() == term]) == 0:
                self.fail(f"Missing TermExtracted event for term {term}.")
            if len([event for event in actual_events_occurrence if event.term.text.lower() == term]) == 0:
                self.fail(f"Missing OccurrenceResolved event for term {term}.")

import asyncio
from typing import AsyncIterable
from unittest import TestCase

from src.knowledge.extract import OpenAIExtractor
from src.terminology.event import TextExtracted, TermExtracted, OccurrenceResolved
from src.terminology.terminology import Blackboard

class TestOpenAIExtractor(TestCase):

    # Before each
    def setUp(self):
        # Initialize the blackboard
        self.blackboard = Blackboard(
            terms=[],
            sources=[]
        )
        # Initialize the KnowledgeSource
        self.extractor = OpenAIExtractor(blackboard=self.blackboard)
        pass

    # After each
    def tearDown(self):
        # Nothing to tear down
        pass

    def test_activate(self):
        """Tests the OpenAIExtractor. It expects an TermExtracted and OccurrenceResolved event
        to be emitted for every term found in the given text."""
        input = """Wenn im Zug außergewöhnliche Sendungen oder außergewöhnliche Fahrzeuge eingestellt sind, müssen sich deren Beförderungsanordnungen beim Zug befinden und die Nummern der Beförderungsanordnungen dem Fahrdienstleiter mitgeteilt worden sein."""
        initial_event = TextExtracted(text=input)
        actual_events = self.collect(self.extractor.activate(initial_event))
        actual_terms = [event.term.text for event in actual_events if type(event) is TermExtracted]

        oracleTerms = ["Zug", "außergewöhnliche Sendung", "außergewöhnliches Fahrzeug", "Beförderungsanordnung", "Nummer der Beförderungsanordnung", "Fahrdienstleiter"]

        print(actual_terms)
        print(oracleTerms)

        # Amount of actual events has to be twice the amount of extracted terms
        # For every term found, there should be
        if len(actual_events) != len(oracleTerms) * 2:
            self.fail(f"Expected {len(oracleTerms) * 2} events, got {len(actual_events)}.\n{actual_events}")

        self.assertEquals(
            len([event for event in actual_events if isinstance(event, TermExtracted)]),
            len([event for event in actual_events if isinstance(event, OccurrenceResolved)]),
            "The amount of TermExtracted events differs from the amount of OccurrenceResolved events."
        )

        # Every event is of type TermExtracted or OccurrenceResolved
        for event in actual_events:
            event_type = type(event)
            if isinstance(event_type, TermExtracted) or isinstance(event_type, OccurrenceResolved):
                self.fail(f"Unexpected event type {event_type}")

        # Every expected term is emitted
        for term in oracleTerms:
            if term not in actual_terms:
                self.fail(f"{term} was not extracted.")

        blackboard_terms_text = [term.text for term in self.blackboard.terms]

        for term in actual_terms:
            # No term other than those expected are extracted
            if term not in oracleTerms:
                self.fail(f"Unexpected term: {term}.")
            # Extracted terms are saved on the blackboard
            if term not in blackboard_terms_text:
                self.fail(f"Term not saved on blackboard: {term}.")

        for event in actual_events:
            # Every OccurrenceResolved event also emits a TermExtracted event
            if type(event) is OccurrenceResolved:
                self.assertIn(event.term.text, actual_terms)
                self.assertIn(event.source, self.blackboard.sources, f"Source not on blackboard {event.source}")



    @staticmethod
    def collect(iterable: AsyncIterable):
        """Synchronously collect all items in the AsyncIterable and return them as a list."""
        async def do():
            return [event async for event in iterable]
        return asyncio.run(do())


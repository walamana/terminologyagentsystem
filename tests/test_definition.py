from unittest import TestCase

from src.knowledge.openai.definition.generator import OpenAIDefinitionGenerator
from src.terminology.event import OccurrenceResolved, PartialDefinitionGenerated
from src.terminology.terminology import Blackboard
from tests.util import collect_async


class TestDefinitionGenerator(TestCase):


    def test_definition_simple(self):
        blackboard = Blackboard(terms=[], sources=[])
        generator = OpenAIDefinitionGenerator(blackboard=blackboard)

        input = "Abstellen\n Züge und Triebfahrzeuge sind abgestellt, wenn sie nicht mit einem Triebfahrzeugführer besetzt sind oder nicht gesteuert werden. Wagen sind abgestellt, sofern sie nicht in Züge eingestellt sind oder nicht rangiert werden.\n Abstoßen\n Abstoßen ist das Bewegen geschobener, nicht mit einem arbeitenden Triebfahrzeug gekuppelter Fahrzeuge durch Beschleunigen, sodass die Fahrzeuge allein weiterfahren, nachdem das Triebfahrzeug angehalten hat."

        term = blackboard.add_term("Abstoßen")
        source = blackboard.add_text_source(input)
        actual_events = collect_async(generator.activate(OccurrenceResolved(term=term, source=source)))

        if len(term.definitions) == 0:
            self.fail("No definition generated for term.")

        expected_definition = "Abstoßen ist das Bewegen geschobener, nicht mit einem arbeitenden Triebfahrzeug gekuppelter Fahrzeuge durch Beschleunigen, sodass die Fahrzeuge allein weiterfahren, nachdem das Triebfahrzeug angehalten hat"

        if expected_definition not in term.definitions[0].text:
            self.fail(f"Expected definition does not match the actual definition.\n"
                      f"Actual: \n{term.definitions[0].text}\n"
                      f"Expected: \n{expected_definition}")

        if len([event for event in actual_events if type(event) is PartialDefinitionGenerated]) == 0:
            self.fail("No PartialDefinitionGenerated event is published.")


    def test_definition_not_enough_context(self):
        blackboard = Blackboard(terms=[], sources=[])
        generator = OpenAIDefinitionGenerator(blackboard=blackboard)

        input = "Dies gilt auch für das Abstoßen, sofern in örtlichen Zusätzen nicht Ausnahmen zugelassen sind."

        term = blackboard.add_term("Abstoßen")
        source = blackboard.add_text_source(input)
        actual_events = collect_async(generator.activate(OccurrenceResolved(term=term, source=source)))

        if len(term.definitions) != 0:
            self.fail(f"A definition was generated, where non should have. Generated Definition: {term.definitions[0].text}")
        if len(actual_events) != 0:
            self.fail("A PartialDefinitionGenerated event was published, where non should have.")
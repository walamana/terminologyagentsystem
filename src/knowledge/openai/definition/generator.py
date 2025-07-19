import asyncio

import numpy as np

from src.knowledge.llm.definition.generator import DEVELOPER_PROMPT, LLMDefinitionGenerator
from src.llm import create_completion_openai
from src.terminology.terminology import OccurrenceResolved, \
    Blackboard


class OpenAIDefinitionGenerator(LLMDefinitionGenerator):

    async def generate_definition_from_source(self, term: str, context: str) -> str | None:
        response, log_probs = await create_completion_openai(
            messages=[
                ("developer", f"{DEVELOPER_PROMPT}"),
                ("user", f"{context}"),
                ("user", f"Definiere den Begriff \"{term}\"."),
            ],
            logprobs=True
        )

        for token in log_probs.content[0].top_logprobs:
            prob = np.exp(token.logprob)
            if token.token == "ERROR" and prob > self.CERTAINTY_THRESHOLD:
                # logger.debug(f"Generation uncertain. Probability of 'ERROR' token {prob}>{self.CERTAINTY_THRESHOLD}!")
                return None

        if response == "ERROR":
            return None
        return response




if __name__ == "__main__":
    blackboard = Blackboard()
    generator = OpenAIDefinitionGenerator(
        blackboard=blackboard
    )

    context = """

Abstellen

Züge und Triebfahrzeuge sind abgestellt, wenn sie nicht mit einem Triebfahrzeugführer besetzt sind oder nicht gesteuert werden. Wagen sind abgestellt, sofern sie nicht in Züge eingestellt sind oder nicht rangiert werden.

Abstoßen

Abstoßen ist das Bewegen geschobener, nicht mit einem arbeitenden Triebfahrzeug gekuppelter Fahrzeuge durch Beschleunigen, so dass die Fahrzeuge allein weiterfahren, nachdem das Triebfahrzeug angehalten hat.

""".strip()

    context = "Dies gilt auch für das Abstoßen, sofern in örtlichen Zusätzen nicht Ausnahmen zugelassen sind."

    term = blackboard.add_term("Abstellen")
    source = blackboard.add_text_source(text=context)

    async def test():
        async for event in generator.activate(OccurrenceResolved(term=term, source=source)):
            print(f"Event {event}")

    asyncio.run(test())
import asyncio
import random
import unittest

import numpy as np

from src.terminology.session import SessionManager, KnowledgeSourcePolicy
from tests.util import create_completion_openai_sync


class TestExtractTerminology(unittest.TestCase):


    def test_multi_word(self):
        input = [
            "Servus Zofia!",
            "Hallo Markus.",
            "Rangiere mir bitte mal den 420er von Gleis 3 auf das Abstellgleis. Passt auf, du musst auf Sicht bis zu den Signalen fahren."
        ]

        expectedTerms = [
            "Rangieren",
            "420",
            "Abstellgleis",
            "auf Sicht fahren",
            "Signal"
        ]

        policy = KnowledgeSourcePolicy(use_llm=True)

        session = SessionManager.create_session(policy)
        blackboard = asyncio.run(
            session.extract_terminology(text=input[len(input) - 1], context="\n".join(input[:-1]))
        )

        terms = [term.text for term in blackboard.terms]
        normalized_terms = [term.normalized_or_text() for term in blackboard.terms]

        print(terms, normalized_terms)

        for term in expectedTerms:
            self.assertIn(term, normalized_terms, f"Expected: {term}")

    def test_multi_word_llm(self):
        input = [
            "Servus Zofia!",
            "Hallo Markus.",
            "Rangiere mir bitte mal den 420er von Gleis 3 auf das Abstellgleis. Passt auf, du musst auf Sicht bis zu den Signalen fahren."
        ]

        expectedTerms = [
            "Rangieren",
            "Abstellgleis",
            "420",
            "auf Sicht fahren",
            "Signal"
        ]

        policy = KnowledgeSourcePolicy(use_llm=True)
        session = SessionManager.create_session(policy)
        blackboard = asyncio.run(
            session.extract_terminology(text=input[len(input) - 1], context="\n".join(input[:-1]))
        )

        normalized_terms = [term.normalized_or_text() for term in blackboard.terms]

        print(expectedTerms, normalized_terms)

        probs_all = []

        for i in range(5):
            print(f"##### TEST {i} #####")
            random.shuffle(expectedTerms)
            random.shuffle(normalized_terms)
            response, logprobs = create_completion_openai_sync(
                messages=[
                    ("user", "Bewerte die Ähnlichkeit der Ergebnisse der Term Extraktion. Gegeben ist ein Ausgangstext, "
                             "aus dem Fachbegriffe extrahiert werden mussten. Der Text ist gegeben. Darunter stehen die erwarteten Begriffe, "
                             "die extrahiert werden sollten. Zum Schluss stehen die tatsächlich extrahierten Begriffe. "
                             "Bewerte die Ähnlichkeit der extrahierten Begriffe. "
                             "Sobald sich ein Begriff grundlegend unterscheidet, beende sofort mit FALSE."
                             "Wenn ein erwarteter Begriff gänzlich fehlt, beende sofort mit FALSE."
                             "Wenn ein Begriff extrahiert wurde, der sicher kein Fachbegriff ist, beende sofort mit FALSE."
                             "Ansonsten Ende sofort mit TRUE."
                             "Bewerte die extrahierten Begriffe."),
                    ("user", f"""{input[len(input) - 1]}\n\nErwartete Begriffe: {", ".join(expectedTerms)}\n\nTatsächliche Begriffe: {", ".join(normalized_terms)}"""),
                ],
                logprobs=True
            )

            print(response)
            probs = {token.token: float(np.exp(token.logprob)) for token in logprobs.content[0].top_logprobs if token.token == "TRUE" or token.token == "FALSE"}
            total_end = sum(probs.values())
            normalized_probs = {token: value / total_end for token, value in probs.items()}
            probs_all.append(normalized_probs["FALSE"])
            print(normalized_probs)
            print("\n\n")
        min_prob = min(probs_all)
        max_prob = max(probs_all)
        avg = sum(probs_all) / len(probs_all)
        var = np.var(probs_all)

        print(f"min: {min_prob}, max: {max_prob}, avg: {avg}, var: {var}")

        self.assertLess(var, 0.05)
        self.assertGreater(avg, 0.8)
        self.assertGreater(min_prob, 0.75)












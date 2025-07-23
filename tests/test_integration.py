import random
from unittest import TestCase

import numpy as np
from fastapi.testclient import TestClient

from src.main import app
from tests.util import create_completion_openai_sync

REDUCE_REASONING = False


class TestIntegrationTerminology(TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def testExtractDomainTerminology(self):
        """This test allows manually added variations"""
        input = [
            "Servus Zofia!",
            "Hallo Markus.",
            "Rangiere mir bitte mal den 420er von Gleis 3 auf das Abstellgleis. Passt auf, du musst auf Sicht bis zu den Signalen fahren."
        ]

        expectedTerms = [
            ["Rangieren"],
            ["420", "420er"],   # The variation "420er" was added after evaluation of the test results, as it is also a valid term
            ["Abstellgleis"],
            ["auf Sicht fahren"],
            ["Signal"]
        ]

        response = self.client.post("/extractTerminology", json={
            "text": input[-1],
            "context": "\n".join(input[:-1])
        })

        response.raise_for_status()

        response = response.json()

        self.assertIn("terms", response)
        terms = [term["normalization"] or term["text"] for term in response["terms"]]
        terms_lower = [term.lower() for term in terms]

        missing_terms = []
        for variations in expectedTerms:
            missing_terms_variation = []
            for term in variations:
                if term.lower() not in terms_lower:
                    missing_terms_variation.append(term)
            # If no variation was matched, the term is not contained in the response
            if len(missing_terms_variation) == len(variations):
                missing_terms.append(variations)

        if len(missing_terms) > 0:
            self.fail(f"Missing the following terms in the response: {missing_terms}")


    def testExtractDomainTerminology_LLM(self):
        input = [
            "Servus Zofia!",
            "Hallo Markus.",
            "Rangiere mir bitte mal den 420er von Gleis 3 auf das Abstellgleis. Passt auf, du musst auf Sicht bis zu den Signalen fahren."
        ]

        expectedTerms = [
            "Rangieren",
            "420",
            "Abstellgleis",
            "Fahrt auf Sicht",
            "Signal"
        ]

        response = self.client.post("/extractTerminology", json={
            "text": input[-1],
            "context": "\n".join(input[:-1])
        })

        response.raise_for_status()

        response = response.json()

        self.assertIn("terms", response)
        terms = [term["normalization"] or term["text"] for term in response["terms"]]


        probs_all = []

        no_reasoning = " sofort" if REDUCE_REASONING else ""

        for i in range(5):
            print(f"##### TEST {i} #####")
            # Note: shuffling the results changed the outcome significantly
            random.shuffle(expectedTerms)
            random.shuffle(terms)
            response, logprobs = create_completion_openai_sync(
                messages=[
                    (
                        "user",
                        "Bewerte die Ähnlichkeit der Ergebnisse der Term Extraktion. Gegeben ist ein Ausgangstext, "
                         "aus dem Fachbegriffe extrahiert werden mussten. Der Text ist gegeben. Darunter stehen die erwarteten Begriffe, "
                         "die extrahiert werden sollten. Zum Schluss stehen die tatsächlich extrahierten Begriffe. "
                         "Bewerte die Ähnlichkeit der extrahierten Begriffe."
                         "Nur sprachliche Variationen für einen erwarteten Begriff sind erlaubt."
                         f"Gibt es für einen erwarteten Begriff keinen ähnlichen extrahierten Begriff, beende{no_reasoning} mit FALSE."
                         f"Wenn ein erwarteter Begriff gänzlich fehlt, beende{no_reasoning} mit FALSE."
                         f"Wenn ein Begriff extrahiert wurde, der sicher kein Fachbegriff ist, beende{no_reasoning} mit FALSE."
                         "Wenn ein Begriff extrahiert wurde, der nicht erwartet wurde, ignoriere diesen. Dies gilt nicht als Unterschied."
                         f"{'Antworte sofort.' if REDUCE_REASONING else ''}"
                         f"Ansonsten Ende{no_reasoning} mit TRUE."
                         "Bewerte die extrahierten Begriffe."
                    ),
                    ("user", f"""{input[len(input) - 1]}\n\nErwartete Begriffe: {", ".join(expectedTerms)}\n\nTatsächliche Begriffe: {", ".join(terms)}"""),
                ],
                logprobs=True
            )

            print(f"{response}")

            # Look at the last 5 output tokens
            last_tokens = logprobs.content[-5:]
            last_tokens.reverse()
            for content in last_tokens:
                cur_token = content.token.strip()
                probs = {token.token: float(np.exp(token.logprob)) for token in content.top_logprobs}
                # print(f"Probs: {probs}")
                if "TRUE" in cur_token or "FALSE" in cur_token:
                    if "TRUE" not in probs.keys() or "FALSE" not in probs.keys():
                        continue
                    probs = {token: prob for token, prob in probs.items() if token == "TRUE" or token == "FALSE"}
                    total_end = sum(probs.values())
                    normalized_probs = {token: value / total_end for token, value in probs.items()}
                    print(normalized_probs)
                    probs_all.append(normalized_probs["TRUE"] if "TRUE" in normalized_probs.keys() else 0)
                    break
            print("")
        min_prob = min(probs_all)
        max_prob = max(probs_all)
        avg = sum(probs_all) / len(probs_all)
        var = np.var(probs_all)

        print(f"min: {min_prob}, max: {max_prob}, avg: {avg}, var: {var}")

        self.assertLess(var, 0.05)
        self.assertGreater(avg, 0.8)
        self.assertGreater(min_prob, 0.75)

from unittest import TestCase

from fastapi.testclient import TestClient

from src.main import app


class TestIntegrationTerminology(TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def testExtractDomainTerminology(self):
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
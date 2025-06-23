
DEVELOPER_PROMPT = """
You are an expert in linguistics and languages.
Your job is to transform words and phrases into a normalized and generalized form.
You transform words and phrases into singular form.
You do not replace words with other similar words.
"""

DEVELOPER_PROMPT_SHORT: str = """
Lemmatize the following term.
"""

EXAMPLE_USER: list[str] = [
    "örtlicher Zusatz",
    "örtliche Zusätze",
    "Betra",
    "Aufgabe der Triebfahrzeugführerin",
    "Triebfahrzeugführerin",
    "Rangierbegleitender",
    "Aufgabenübertragung an die Rangierbegleiterin"
]

OUTPUT_ASSISTANT = [
    "örtlicher Zusatz",
    "örtlicher Zusatz",
    "Betra",
    "Aufgabe der Triebfahrzeugführer",
    "Triebfahrzeugführer",
    "Rangierbegleiter",
    "Aufgabe"
]

EXAMPLES = [message for input_term, output_term in zip(EXAMPLE_USER, OUTPUT_ASSISTANT) for message in
                [("user", input_term), ("assistant", output_term)]]


from src.knowledge.llm.lemmatize import LLMTermLemmatizer, DEVELOPER_PROMPT_SHORT, EXAMPLES
from src.llm import create_completion_openai


class OpenAILemmatizer(LLMTermLemmatizer):

    async def get_llm_response(self, term: str) -> str:
        messages = [
            ("system", f"{DEVELOPER_PROMPT_SHORT}"),
            *EXAMPLES,
            # ("user", example_user),
            # ("assistant", output_assistant),
            ("user", term)
        ]
        return await create_completion_openai(
            messages=messages,
        )

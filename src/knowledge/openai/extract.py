from src.knowledge.llm.extract import LLMTermExtractor, DEVELOPER_PROMPT, EXAMPLE_USER, OUTPUT_ASSISTANT
from src.llm import create_completion_openai


class OpenAIExtractor(LLMTermExtractor):

    async def get_llm_response(self, text: str) -> str:
        return await create_completion_openai(
            messages=[
                ("developer", f"{DEVELOPER_PROMPT}"),
                ("user", EXAMPLE_USER),
                ("assistant", OUTPUT_ASSISTANT),
                ("user", "Input: \n" + text)
            ]
        )
import os
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

class LLMService:
    def __init__(self, temperature: float = 0.2) -> None:
        self._llm = ChatOpenAI(
            model=os.environ["FACTCHAT_MODEL"],
            temperature=temperature,
            base_url=os.environ["FACTCHAT_BASE_URL"].rstrip("/"),
            api_key=os.environ["FACTCHAT_API_KEY"],
        )

    def invoke(self, messages: list[BaseMessage]) -> str:
        response = self._llm.invoke(messages)
        content = response.content
        if isinstance(content, str):
            return content
        return "".join(str(part) for part in content)


if __name__ == "__main__":
    # 프로젝트 루트에서, .env에 FACTCHAT_* 설정 후:
    # python -m src.services.llm_service
    from dotenv import load_dotenv
    from langchain_core.messages import HumanMessage, SystemMessage

    load_dotenv()

    llm = LLMService()
    text = llm.invoke(
        [
            SystemMessage(content="짧게 한 문장으로만 답하세요."),
            HumanMessage(content="1+1은?"),
        ]
    )
    print(text)

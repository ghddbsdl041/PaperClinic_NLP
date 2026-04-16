from langchain_core.messages import HumanMessage, SystemMessage
from src.services.llm_service import LLMService
from src.utils.prompt_manager import PromptManager


class QuestionService:
    def __init__(self, llm: LLMService, prompts: PromptManager) -> None:
        self._llm = llm
        self._prompts = prompts

    def generate_question(self, context: str) -> str:
        user = self._prompts.render_user("question_generation", context=context)
        system = self._prompts.system_prompt("question_generation")
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]
        return self._llm.invoke(messages).strip()


if __name__ == "__main__":
    # 프로젝트 루트에서, .env에 OPENAI_API_KEY·FACTCHAT_* 설정 후:
    # python -m src.services.question_service
    from dotenv import load_dotenv

    load_dotenv()

    root = Path(__file__).resolve().parents[2]
    cfg = root / "config" / "prompts.yaml"

    llm = LLMService()
    prompts = PromptManager(cfg)
    svc = QuestionService(llm, prompts)

    dummy_context = "이 수업은 자연어 처리 파이프라인 설계와 평가를 다룬다."
    q = svc.generate_question(dummy_context)
    print("생성 질문:", q)

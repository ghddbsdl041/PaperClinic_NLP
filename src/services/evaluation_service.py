import json
import re
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage
from src.services.llm_service import LLMService
from src.utils.prompt_manager import PromptManager
from src.utils.yaml_loader import load_yaml


def _parse_json_from_llm(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    return json.loads(text)


class EvaluationService:
    def __init__(
        self,
        llm: LLMService,
        prompts: PromptManager,
        evaluation_config_path: Path | str,
    ) -> None:
        self._llm = llm
        self._prompts = prompts
        self._eval_cfg = load_yaml(evaluation_config_path)

    def _criteria_block(self) -> str:
        lines: list[str] = []
        for name, meta in self._eval_cfg["criteria"].items():
            lines.append(f"- {name} (weight={meta['weight']}): {meta['description']}")
        return "\n".join(lines)

    def _max_score(self) -> int:
        return int(self._eval_cfg["score_policy"]["max_score"])

    def evaluate(
        self,
        *,
        question: str,
        user_answer: str,
        context: str,
    ) -> dict:
        user = self._prompts.render_user(
            "evaluation",
            criteria_block=self._criteria_block(),
            max_score=str(self._max_score()),
            question=question,
            user_answer=user_answer,
            context=context,
        )
        system = self._prompts.system_prompt("evaluation")
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]
        raw = self._llm.invoke(messages)
        return _parse_json_from_llm(raw)


if __name__ == "__main__":
    # 프로젝트 루트에서, .env 설정 후:
    # python -m src.services.evaluation_service
    from dotenv import load_dotenv

    load_dotenv()

    root = Path(__file__).resolve().parents[2]
    prompts_path = root / "config" / "prompts.yaml"
    eval_path = root / "config" / "evaluation.yaml"

    llm = LLMService()
    prompts = PromptManager(prompts_path)
    svc = EvaluationService(llm, prompts, eval_path)

    out = svc.evaluate(
        question="이 수업의 목표는?",
        user_answer="자연어 시스템을 설계하고 평가하는 능력을 기른다.",
        context="[chunk 1]\n이 수업은 NLP 시스템 설계와 평가를 다룬다.",
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))

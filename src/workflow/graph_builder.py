from pathlib import Path

from langchain_core.documents import Document
from src.workflow.state import WorkflowState
from langgraph.graph.state import CompiledStateGraph
from src.services.document_service import DocumentService
from src.services.evaluation_service import EvaluationService
from src.services.llm_service import LLMService
from src.services.question_service import QuestionService
from src.services.retrieval_service import RetrievalService
from src.utils.prompt_manager import PromptManager
from src.workflow.state import WorkflowState
from langgraph.graph import END, StateGraph


_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def _format_context(docs: list[Document]) -> str:
    parts: list[str] = []
    for i, d in enumerate(docs, start=1):
        parts.append(f"[chunk {i}]\n{d.page_content.strip()}")
    return "\n\n".join(parts)


class WorkflowBuilder:
    """LangGraph 조립 (실습에서 구현)."""

    def __init__(self) -> None:
        self._documents = DocumentService()
        self._retrieval = RetrievalService()
        self._llm = LLMService()
        self._prompts = PromptManager(_CONFIG_DIR / "prompts.yaml")
        self._questions = QuestionService(self._llm, self._prompts)
        self.evaluation = EvaluationService(
            self._llm,
            self._prompts,
            _CONFIG_DIR / "evaluation.yaml",
        )

    def _load_docs(self, state: WorkflowState) -> WorkflowState:
        docs = self._documents.load(state["file_path"])
        return {"raw_documents": docs}

    def _split_docs(self, state: WorkflowState) -> WorkflowState:
        splits = self._documents.split(state["raw_documents"])
        return {"splits": splits}

    def _build_store(self, state: WorkflowState) -> WorkflowState:
        store = self._retrieval.build_store(state["splits"])
        return {"vector_store": store}

    def _generate_question(self, state: WorkflowState) -> WorkflowState:
        splits = state["splits"]
        seed = _format_context(splits[: min(6, len(splits))])
        text = self._questions.generate_question(seed)
        return {"generated_question": text}

    def _retrieve_context(self, state: WorkflowState) -> WorkflowState:
        hits = self._retrieval.retrieve(
            state["vector_store"],
            state["generated_question"],
            k=4,
        )
        return {"retrieved_context": _format_context(hits)}

    def build(self) -> CompiledStateGraph:
        graph = StateGraph(WorkflowState)
        graph.add_node("load_docs", self._load_docs)
        graph.add_node("split_docs", self._split_docs)
        graph.add_node("build_store", self._build_store)
        graph.add_node("generate_question", self._generate_question)
        graph.add_node("retrieve_context", self._retrieve_context)
        graph.set_entry_point("load_docs")
        graph.add_edge("load_docs", "split_docs")
        graph.add_edge("split_docs", "build_store")
        graph.add_edge("build_store", "generate_question")
        graph.add_edge("generate_question", "retrieve_context")
        graph.add_edge("retrieve_context", END)
        return graph.compile()

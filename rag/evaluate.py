import json
from typing import Any, Dict, List, Union

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from tqdm import tqdm


class RAGEvaluator:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def _ensure_list(
        self, data_or_path: Union[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        if isinstance(data_or_path, str):
            with open(data_or_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, list):
                raise ValueError("Input JSON must be a list of objects")
            return loaded
        return data_or_path

    def _judge_relevance(self, query: str, chunk_text: str) -> int:
        
        template = """You are a strict relevance judge.
            Given a search query and one candidate text chunk, respond with exactly one character: 1 if the chunk contains information relevant for answering the query, otherwise 0.
            No explanation and no extra characters.
            Query: {query}
            Chunk: {chunk_text}
            Answer (0 or 1):"""
        prompt = ChatPromptTemplate.from_template(template)
        model = OllamaLLM(model="gemma3:1b")
        chain = prompt | model
        try:
            result = chain.invoke({"query": query, "chunk_text": chunk_text})
            response = result.strip()
            return 1 if response.startswith("1") else 0
        except Exception:
            return 0

    def evaluate(
        self, data_or_path: Union[str, List[Dict[str, Any]]]
    ) -> Dict[str, Dict[str, int]]:
        items = self._ensure_list(data_or_path)
        evaluations: Dict[str, Dict[str, int]] = {}

        for obj in tqdm(items, desc="Evaluating queries", unit="query"):
            query = obj.get("query", "").strip()
            query_id = str(obj.get("query_num", len(evaluations) + 1))
            chunk_scores: Dict[str, int] = {}

            # chunk_indices: List[int] = []
            # for key in obj.keys():
            #     if key.startswith("chunk_") and key.endswith("_text"):
            #         try:
            #             idx = int(key.split("_")[1])
            #             chunk_indices.append(idx)
            #         except Exception:
            #             continue
            # chunk_indices.sort()

            for idx in range(1, 6):
                text_key = f"chunk_{idx}_text"
                id_key = f"chunk_{idx}_chunk_id"
                file_key = f"chunk_{idx}_filename"

                chunk_text = str(obj.get(text_key, ""))
                if not chunk_text.strip():
                    continue
                chunk_identifier = (
                    str(obj.get(id_key))
                    if obj.get(id_key) is not None
                    else (
                        str(obj.get(file_key))
                        if obj.get(file_key) is not None
                        else f"chunk_{idx}"
                    )
                )
                # FIXME: Judge relevance function not working as expected
                relevance = self._judge_relevance(query=query, chunk_text=chunk_text)
                print(query_id, "\t", chunk_identifier, "\t", relevance)
                chunk_scores[chunk_identifier] = relevance

            evaluations[query_id] = chunk_scores

        return evaluations

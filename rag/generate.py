import json
from typing import Any, Dict, List, Optional, Union

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from tqdm import tqdm


class RAGGenerator:
    """
    Generate answers for queries using retrieved text chunks and Ollama LLM.
    
    Takes retrieval results (query + retrieved chunks) and generates comprehensive
    answers by synthesizing information from the relevant text chunks.
    """
    
    def __init__(self, model_name: str = "llama3.1:8b", temperature: float = 0.3):
        """
        Initialize the RAG Generator.
        
        Args:
            model_name: Name of the Ollama model to use for generation
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.model = OllamaLLM(model=model_name, temperature=temperature)
        print(f"✅ Initialized RAG Generator with model: {model_name}")
    
    def _ensure_list(
        self, data_or_path: Union[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Load data from file path or return list as-is."""
        if isinstance(data_or_path, str):
            with open(data_or_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, list):
                raise ValueError("Input JSON must be a list of objects")
            return loaded
        return data_or_path
    
    def _extract_chunks(self, obj: Dict[str, Any], max_chunks: int = 5) -> List[Dict[str, str]]:
        """
        Extract text chunks from a retrieval result object.
        
        Args:
            obj: Retrieval result object containing query and chunks
            max_chunks: Maximum number of chunks to extract
            
        Returns:
            List of dicts with chunk text, filename, and chunk_id
        """
        chunks = []
        
        for idx in range(1, max_chunks + 1):
            text_key = f"chunk_{idx}_text"
            file_key = f"chunk_{idx}_filename"
            id_key = f"chunk_{idx}_chunk_id"
            score_key = f"chunk_{idx}_score"
            
            chunk_text = obj.get(text_key, "").strip()
            if not chunk_text:
                continue
            
            chunks.append({
                "text": chunk_text,
                "filename": obj.get(file_key, f"unknown_file_{idx}"),
                "chunk_id": obj.get(id_key, f"unknown_id_{idx}"),
                "score": obj.get(score_key, 0.0),
            })
        
        return chunks
    
    def _generate_answer(self, query: str, chunks: List[Dict[str, str]]) -> str:
        """
        Generate an answer for a query based on retrieved chunks.
        
        Args:
            query: The user's query
            chunks: List of retrieved text chunks with metadata
            
        Returns:
            Generated answer string
        """
        if not chunks:
            return "No relevant information found to answer this query."
        
        # Prepare context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[Source {i} - {chunk['filename']}]")
            context_parts.append(chunk['text'])
            context_parts.append("")  # Empty line for readability
        
        context = "\n".join(context_parts)
        
        # Create prompt template
        template = """You are a helpful AI assistant that answers questions based on provided context.

Your task:
1. Carefully read the context from multiple sources below
2. Answer the user's question using ONLY information from the provided context
3. If the context contains relevant information, provide a comprehensive and well-structured answer
4. Cite sources by mentioning the source number (e.g., "According to Source 1...")
5. If the context doesn't contain enough information to fully answer the question, acknowledge this
6. Do not make up information or use knowledge outside the provided context

Context from retrieved documents:
{context}

User Question: {query}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.model
        
        try:
            response = chain.invoke({
                "query": query,
                "context": context
            })
            return response.strip()
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def generate_single(self, query: str, chunks: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Generate answer for a single query with its chunks.
        
        Args:
            query: The query string
            chunks: List of chunk dictionaries
            
        Returns:
            Dict with query, answer, and metadata
        """
        answer = self._generate_answer(query, chunks)
        
        return {
            "query": query,
            "answer": answer,
            "num_chunks_used": len(chunks),
            "sources": [chunk["filename"] for chunk in chunks],
        }
    
    def generate(
        self, 
        data_or_path: Union[str, List[Dict[str, Any]]],
        output_path: Optional[str] = None,
        max_chunks: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate answers for all queries in the retrieval results.
        
        Args:
            data_or_path: Path to retrieval results JSON or list of result objects
            output_path: Optional path to save generated answers as JSON
            max_chunks: Maximum number of chunks to use per query
            
        Returns:
            List of dicts with query, answer, and metadata
        """
        items = self._ensure_list(data_or_path)
        results = []
        
        print(f"🤖 Generating answers for {len(items)} queries using {self.model_name}...")
        
        for obj in tqdm(items, desc="Generating answers", unit="query"):
            query = obj.get("query", "").strip()
            query_num = obj.get("query_num", len(results) + 1)
            
            if not query:
                print(f"⚠️  Skipping query {query_num}: empty query text")
                continue
            
            # Extract chunks
            chunks = self._extract_chunks(obj, max_chunks=max_chunks)
            
            if not chunks:
                print(f"⚠️  No chunks found for query {query_num}")
                answer = "No relevant information found to answer this query."
                results.append({
                    "query_num": query_num,
                    "query": query,
                    "answer": answer,
                    "num_chunks_used": 0,
                    "sources": [],
                })
                continue
            
            # Generate answer
            answer = self._generate_answer(query, chunks)
            
            results.append({
                "query_num": query_num,
                "query": query,
                "answer": answer,
                "num_chunks_used": len(chunks),
                "sources": [chunk["filename"] for chunk in chunks],
                "chunk_scores": [chunk["score"] for chunk in chunks],
            })
        
        # Save to file if output path provided
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"✅ Generated answers saved to: {output_path}")
        
        return results
    
    def generate_streaming(
        self,
        query: str,
        chunks: List[Dict[str, str]],
    ):
        """
        Generate answer with streaming output (for real-time display).
        
        Args:
            query: The query string
            chunks: List of chunk dictionaries
            
        Yields:
            Chunks of generated text as they become available
        """
        if not chunks:
            yield "No relevant information found to answer this query."
            return
        
        # Prepare context
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[Source {i} - {chunk['filename']}]")
            context_parts.append(chunk['text'])
            context_parts.append("")
        
        context = "\n".join(context_parts)
        
        template = """You are a helpful AI assistant that answers questions based on provided context.

Your task:
1. Carefully read the context from multiple sources below
2. Answer the user's question using ONLY information from the provided context
3. Provide a comprehensive and well-structured answer
4. Cite sources by mentioning the source number
5. If the context doesn't contain enough information, acknowledge this

Context from retrieved documents:
{context}

User Question: {query}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Use streaming-enabled model
        streaming_model = OllamaLLM(
            model=self.model_name,
            temperature=self.temperature
        )
        
        chain = prompt | streaming_model
        
        try:
            for chunk in chain.stream({"query": query, "context": context}):
                yield chunk
        except Exception as e:
            yield f"Error generating answer: {str(e)}"


def generate_single_answer(
    retrieval_result: Dict[str, Any],
    model_name: str = "llama3.1:8b",
    temperature: float = 0.3,
    max_chunks: int = 5
) -> Dict[str, Any]:
    """
    Generate answer for a single query from its retrieval result.
    
    Args:
        retrieval_result: Dict containing query and retrieved chunks
                         (output from single_retrieval or process_single_query)
        model_name: Ollama model to use
        temperature: Sampling temperature
        max_chunks: Maximum chunks to use
        
    Returns:
        Dict with query, answer, sources, and metadata
    """
    generator = RAGGenerator(model_name=model_name, temperature=temperature)
    
    # Extract query
    query = retrieval_result.get("query", "").strip()
    query_num = retrieval_result.get("query_num", "1")
    
    if not query:
        return {
            "query_num": query_num,
            "query": query,
            "answer": "Empty query provided.",
            "num_chunks_used": 0,
            "sources": [],
        }
    
    # Extract chunks from retrieval result
    chunks = generator._extract_chunks(retrieval_result, max_chunks=max_chunks)
    
    if not chunks:
        return {
            "query_num": query_num,
            "query": query,
            "answer": "No relevant information found to answer this query.",
            "num_chunks_used": 0,
            "sources": [],
        }
    
    # Generate answer
    answer = generator._generate_answer(query, chunks)
    
    return {
        "query_num": query_num,
        "query": query,
        "answer": answer,
        "num_chunks_used": len(chunks),
        "sources": [chunk["filename"] for chunk in chunks],
        "chunk_scores": [chunk["score"] for chunk in chunks],
    }


def generate_answers(
    retrieval_results_path: str,
    output_path: Optional[str] = None,
    model_name: str = "llama3.1:8b",
    temperature: float = 0.3,
    max_chunks: int = 5
) -> List[Dict[str, Any]]:
    """
    Convenience function to generate answers from retrieval results.
    
    Args:
        retrieval_results_path: Path to JSON file with retrieval results
        output_path: Path to save generated answers (optional)
        model_name: Ollama model to use
        temperature: Sampling temperature
        max_chunks: Maximum chunks to use per query
        
    Returns:
        List of dicts with queries and generated answers
    """
    generator = RAGGenerator(model_name=model_name, temperature=temperature)
    return generator.generate(
        retrieval_results_path,
        output_path=output_path,
        max_chunks=max_chunks
    )

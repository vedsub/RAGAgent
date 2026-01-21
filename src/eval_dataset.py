"""
Synthetic Dataset Generator for RAG Evaluation

Generates query-document pairs from existing documents using LLM,
creating ground truth mappings for retrieval evaluation.
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from dotenv import load_dotenv

from .observability import get_logger
from .llm_observability import get_llm_wrapper

load_dotenv()


@dataclass
class EvalSample:
    """Single evaluation sample with query and ground truth"""

    query: str
    relevant_doc_ids: List[str]
    source_chunk: str
    source_metadata: Dict[str, Any]


@dataclass
class EvalDataset:
    """Complete evaluation dataset"""

    samples: List[EvalSample]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict:
        return {"samples": [asdict(s) for s in self.samples], "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: Dict) -> "EvalDataset":
        samples = [EvalSample(**s) for s in data["samples"]]
        return cls(samples=samples, metadata=data["metadata"])


class SyntheticDatasetGenerator:
    """Generates synthetic evaluation dataset from documents using LLM"""

    def __init__(self, documents: List[Document] = None):
        """
        Initialize generator

        Args:
            documents: List of LangChain Document objects
        """
        self.documents = documents or []

        # Initialize Groq LLM
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")

        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.7,  # Higher temp for diverse queries
            max_tokens=512,
        )

        self.logger = get_logger()
        self.llm_wrapper = get_llm_wrapper()
        self.obs_manager = self.llm_wrapper.obs_manager

        # Prompt for query generation
        self.query_prompt = """You are a helpful assistant that generates search queries.

Given the following text chunk from a document, generate {num_queries} diverse search queries that a user might ask to find this information. The queries should:
1. Be natural questions or search phrases
2. Be answerable by the given text
3. Vary in style (some questions, some keyword searches)
4. Cover different aspects of the content

Text chunk:
---
{chunk}
---

Generate exactly {num_queries} queries, one per line. Output ONLY the queries, nothing else."""

    def set_documents(self, documents: List[Document]):
        """Set documents for generation"""
        self.documents = documents

    def _assign_doc_ids(self) -> Dict[int, str]:
        """Assign unique IDs to documents"""
        return {i: f"doc_{i}" for i in range(len(self.documents))}

    def _generate_queries_for_chunk(
        self, chunk: str, num_queries: int = 2
    ) -> List[str]:
        """
        Generate queries for a document chunk using LLM

        Args:
            chunk: Text content to generate queries for
            num_queries: Number of queries to generate

        Returns:
            List of generated queries
        """

        # Add observability for synthetic query generation
        @self.llm_wrapper.observe_llm_synthetic_generation(
            model_name=self.llm.model_name
        )
        def _generate_queries():
            prompt = self.query_prompt.format(
                chunk=chunk[:2000], num_queries=num_queries
            )

            return self.llm.invoke(
                [
                    SystemMessage(
                        content="You are a search query generator. Output only the queries, one per line."
                    ),
                    HumanMessage(content=prompt),
                ]
            )

        try:
            self.logger.info(
                "Generating synthetic queries",
                chunk_length=len(chunk),
                target_queries=num_queries,
            )

            response = _generate_queries()

            # Parse queries from response
            queries = [
                q.strip()
                for q in response.content.strip().split("\n")
                if q.strip() and not q.strip().startswith(("-", "*", "•"))
            ]

            # Clean up numbered queries (e.g., "1. Query" -> "Query")
            cleaned_queries = []
            for q in queries:
                # Remove leading numbers like "1.", "1)", "1:"
                import re

                cleaned = re.sub(r"^\d+[\.\)\:]\s*", "", q)
                if cleaned:
                    cleaned_queries.append(cleaned)

            final_queries = cleaned_queries[:num_queries]

            self.logger.info(
                "Synthetic queries generated",
                requested=num_queries,
                generated=len(final_queries),
                response_length=len(response.content),
            )

            return final_queries

        except Exception as e:
            self.logger.error(
                "Synthetic query generation failed",
                chunk_length=len(chunk),
                target_queries=num_queries,
                error=str(e),
                exc_info=True,
            )
            return []

    def build_eval_dataset(
        self,
        num_samples: int = 50,
        queries_per_chunk: int = 2,
        min_chunk_length: int = 100,
    ) -> EvalDataset:
        """
        Build evaluation dataset from documents

        Args:
            num_samples: Target number of samples to generate
            queries_per_chunk: Number of queries per document chunk
            min_chunk_length: Minimum chunk length to consider

        Returns:
            EvalDataset with generated samples
        """
        if not self.documents:
            raise ValueError("No documents loaded. Call set_documents() first.")

        doc_ids = self._assign_doc_ids()

        # Filter documents by minimum length
        valid_docs = [
            (i, doc)
            for i, doc in enumerate(self.documents)
            if len(doc.page_content) >= min_chunk_length
        ]

        if not valid_docs:
            raise ValueError("No documents meet minimum length requirement")

        print(f"Building dataset from {len(valid_docs)} valid documents...")

        # Calculate how many documents to sample
        docs_needed = min(num_samples // queries_per_chunk + 1, len(valid_docs))
        sampled_docs = random.sample(valid_docs, docs_needed)

        samples = []
        for idx, doc in sampled_docs:
            doc_id = doc_ids[idx]
            chunk = doc.page_content

            # Generate queries
            queries = self._generate_queries_for_chunk(chunk, queries_per_chunk)

            for query in queries:
                sample = EvalSample(
                    query=query,
                    relevant_doc_ids=[doc_id],
                    source_chunk=chunk[:500],  # Truncate for storage
                    source_metadata=doc.metadata,
                )
                samples.append(sample)

                if len(samples) >= num_samples:
                    break

            if len(samples) >= num_samples:
                break

            # Progress indicator
            if len(samples) % 10 == 0:
                print(f"Generated {len(samples)}/{num_samples} samples...")

        dataset = EvalDataset(
            samples=samples,
            metadata={
                "num_samples": len(samples),
                "num_documents": len(self.documents),
                "queries_per_chunk": queries_per_chunk,
                "doc_id_mapping": doc_ids,
            },
        )

        print(f"Dataset created with {len(samples)} samples")
        return dataset

    def save_dataset(self, dataset: EvalDataset, path: str):
        """Save dataset to JSON file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(dataset.to_dict(), f, indent=2)

        print(f"Dataset saved to {path}")

    def load_dataset(self, path: str) -> EvalDataset:
        """Load dataset from JSON file"""
        with open(path, "r") as f:
            data = json.load(f)

        dataset = EvalDataset.from_dict(data)
        print(f"Loaded dataset with {len(dataset.samples)} samples")
        return dataset


if __name__ == "__main__":
    # Example usage
    from src.data_loader import load_all_documents

    docs = load_all_documents("data")
    generator = SyntheticDatasetGenerator(documents=docs)

    dataset = generator.build_eval_dataset(num_samples=10, queries_per_chunk=2)
    generator.save_dataset(dataset, "eval_data/eval_dataset.json")

    # Print sample
    if dataset.samples:
        sample = dataset.samples[0]
        print(f"\nSample query: {sample.query}")
        print(f"Relevant docs: {sample.relevant_doc_ids}")

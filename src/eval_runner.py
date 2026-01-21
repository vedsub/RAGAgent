"""
Evaluation Runner

Main orchestrator for running RAG evaluations:
- Loads/generates synthetic dataset
- Runs retrieval on vector store
- Computes all metrics
- Generates evaluation report
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from src.eval_dataset import SyntheticDatasetGenerator, EvalDataset
from src.eval_metrics import RetrievalMetrics, EvalResults
from src.eval_judge import LLMJudge, JudgeScore
from src.vectorstore import FaissVectorStore
from src.data_loader import load_all_documents
from src.observability import get_logger, get_tracer
from src.llm_observability import get_llm_wrapper


@dataclass
class EvalReport:
    """Complete evaluation report"""

    timestamp: str
    num_queries: int
    k_values: List[int]

    # Retrieval metrics (averaged across queries)
    retrieval_metrics: Dict[str, Any]

    # LLM judge metrics
    judge_metrics: Dict[str, Any]

    # Per-query breakdown (optional, for detailed analysis)
    per_query_results: Optional[List[Dict]] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    def summary(self) -> str:
        """Generate human-readable summary"""
        lines = [
            "=" * 60,
            "RAG EVALUATION REPORT",
            "=" * 60,
            f"Timestamp: {self.timestamp}",
            f"Queries evaluated: {self.num_queries}",
            f"K values: {self.k_values}",
            "",
            "RETRIEVAL METRICS",
            "-" * 40,
            f"MRR: {self.retrieval_metrics.get('mrr', 0):.4f}",
        ]

        for k in self.k_values:
            p = self.retrieval_metrics.get("precision", {}).get(str(k), 0)
            r = self.retrieval_metrics.get("recall", {}).get(str(k), 0)
            n = self.retrieval_metrics.get("ndcg", {}).get(str(k), 0)
            lines.append(f"  @{k}: Precision={p:.4f}, Recall={r:.4f}, NDCG={n:.4f}")

        lines.extend(
            [
                "",
                "CONTEXT RELEVANCE (LLM Judge)",
                "-" * 40,
                f"Average Score: {self.judge_metrics.get('average_score', 0):.2f}/5",
                f"Score Distribution: {self.judge_metrics.get('distribution', {})}",
                "=" * 60,
            ]
        )

        return "\n".join(lines)


class EvalRunner:
    """Main evaluation orchestrator"""

    def __init__(
        self,
        vector_store: FaissVectorStore = None,
        data_dir: str = "data",
        eval_data_dir: str = "eval_data",
    ):
        """
        Initialize evaluation runner

        Args:
            vector_store: Pre-loaded vector store (optional)
            data_dir: Directory containing source documents
            eval_data_dir: Directory for evaluation data/results
        """
        self.vector_store = vector_store
        self.data_dir = data_dir
        self.eval_data_dir = Path(eval_data_dir)
        self.eval_data_dir.mkdir(parents=True, exist_ok=True)

        self.dataset: Optional[EvalDataset] = None
        self.judge = None  # Lazy init
        self.logger = get_logger()
        self.tracer = get_tracer()
        self.llm_wrapper = get_llm_wrapper()
        self.obs_manager = self.llm_wrapper.obs_manager

    def load_or_create_vector_store(
        self, store_path: str = "faiss_store"
    ) -> FaissVectorStore:
        """Load existing vector store or create new one"""
        store = FaissVectorStore(store_path)
        store.load()

        if store.index is None:
            print("No existing store found, building from documents...")
            docs = load_all_documents(self.data_dir)
            store.build_from_documents(docs)
            store.save()

        self.vector_store = store
        return store

    def generate_dataset(self, num_samples: int = 50, save: bool = True) -> EvalDataset:
        """Generate synthetic evaluation dataset"""
        print(f"Generating synthetic dataset with {num_samples} samples...")

        docs = load_all_documents(self.data_dir)
        generator = SyntheticDatasetGenerator(documents=docs)

        self.dataset = generator.build_eval_dataset(
            num_samples=num_samples, queries_per_chunk=2
        )

        if save:
            dataset_path = self.eval_data_dir / "eval_dataset.json"
            generator.save_dataset(self.dataset, str(dataset_path))

        return self.dataset

    def load_dataset(self, path: str = None) -> EvalDataset:
        """Load existing evaluation dataset"""
        if path is None:
            path = self.eval_data_dir / "eval_dataset.json"

        generator = SyntheticDatasetGenerator()
        self.dataset = generator.load_dataset(str(path))
        return self.dataset

    def run_evaluation(
        self, k_values: List[int] = None, run_judge: bool = True, verbose: bool = True
    ) -> EvalReport:
        """
        Run full evaluation

        Args:
            k_values: K values for metrics (default: [1, 3, 5, 10])
            run_judge: Whether to run LLM judge scoring
            verbose: Print progress

        Returns:
            EvalReport with all results
        """
        # Start workflow-level tracing
        if self.tracer:
            span = self.tracer.start_span("evaluation.run_full_evaluation")

        start_time = datetime.now()

        if k_values is None:
            k_values = [1, 3, 5, 10]

        if self.vector_store is None:
            self.load_or_create_vector_store()

        if self.dataset is None:
            # Try to load existing dataset
            try:
                self.load_dataset()
            except FileNotFoundError:
                self.logger.info("No dataset found. Generating new one...")
                self.generate_dataset(num_samples=50)

        self.logger.info(
            "Starting evaluation workflow",
            num_queries=len(self.dataset.samples),
            k_values=k_values,
            run_judge=run_judge,
        )

        if verbose:
            print(f"Running evaluation on {len(self.dataset.samples)} queries...")

        # Get doc ID mapping from dataset
        doc_id_mapping = self.dataset.metadata.get("doc_id_mapping", {})
        # Reverse mapping: doc_id -> index
        id_to_idx = {v: int(k) for k, v in doc_id_mapping.items()}

        # Run retrieval and compute metrics
        all_results = []
        judge_pairs = []
        per_query_data = []

        max_k = max(k_values)

        for i, sample in enumerate(self.dataset.samples):
            query = sample.query
            relevant_ids = set(sample.relevant_doc_ids)

            # Run retrieval with observability
            with self.llm_wrapper.observe_vector_store_operation(
                "evaluation_query", query_index=i, top_k=max_k
            ) as vec_ctx:
                results = self.vector_store.query(query, top_k=max_k)
                retrieved_ids = [r["id"] for r in results]
                vec_ctx["results_count"] = len(retrieved_ids)

            # Compute metrics for this query
            query_results = RetrievalMetrics.compute_all(
                retrieved=retrieved_ids, relevant=relevant_ids, k_values=k_values
            )
            all_results.append(query_results)

            # Collect for judge evaluation (use top retrieved context)
            if run_judge and results:
                top_context = results[0]["content"]
                judge_pairs.append((query, top_context))

            # Store per-query data
            per_query_data.append(
                {
                    "query": query,
                    "relevant_ids": list(relevant_ids),
                    "retrieved_ids": retrieved_ids,
                    "metrics": query_results.to_dict(),
                }
            )

            if verbose and (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(self.dataset.samples)} queries...")

        # Aggregate retrieval metrics
        avg_results = RetrievalMetrics.aggregate_results(all_results)

        # Convert keys to strings for JSON serialization
        retrieval_metrics = {
            "mrr": avg_results.mrr,
            "precision": {str(k): v for k, v in avg_results.precision.items()},
            "recall": {str(k): v for k, v in avg_results.recall.items()},
            "ndcg": {str(k): v for k, v in avg_results.ndcg.items()},
        }

        # Run LLM judge with observability
        judge_metrics = {"average_score": 0, "distribution": {}, "num_evaluated": 0}

        if run_judge and judge_pairs:
            if self.tracer:
                judge_span = self.tracer.start_span("evaluation.llm_judge_scoring")

            if verbose:
                print("\nRunning LLM judge evaluation...")

            if self.judge is None:
                self.judge = LLMJudge()

            self.logger.info(
                "Starting LLM judge evaluation", num_pairs=len(judge_pairs)
            )

            scores = self.judge.batch_score(judge_pairs, verbose=verbose)

            judge_metrics = {
                "average_score": self.judge.compute_average_score(scores),
                "distribution": self.judge.get_score_distribution(scores),
                "num_evaluated": len(scores),
            }

            # Add judge scores to per-query data
            for i, score in enumerate(scores):
                if i < len(per_query_data):
                    per_query_data[i]["judge_score"] = score.score
                    per_query_data[i]["judge_reasoning"] = score.reasoning

            if self.tracer and judge_span:
                judge_span.end()

        # Create report
        report = EvalReport(
            timestamp=datetime.now().isoformat(),
            num_queries=len(self.dataset.samples),
            k_values=k_values,
            retrieval_metrics=retrieval_metrics,
            judge_metrics=judge_metrics,
            per_query_results=per_query_data,
        )

        total_time = (datetime.now() - start_time).total_seconds()

        self.logger.info(
            "Evaluation workflow completed",
            total_queries=len(self.dataset.samples),
            total_time_seconds=total_time,
            avg_mrr=avg_results.mrr,
            judge_avg_score=judge_metrics.get("average_score", 0),
        )

        if self.tracer and span:
            span.set_attribute("evaluation.total_queries", len(self.dataset.samples))
            span.set_attribute("evaluation.total_time_seconds", total_time)
            span.set_attribute("evaluation.avg_mrr", avg_results.mrr)
            span.set_attribute(
                "evaluation.judge_avg_score", judge_metrics.get("average_score", 0)
            )
            span.end()

        return report

    def save_report(self, report: EvalReport, filename: str = None):
        """Save evaluation report to JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eval_report_{timestamp}.json"

        path = self.eval_data_dir / filename

        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        print(f"Report saved to {path}")
        return path

    def print_report(self, report: EvalReport):
        """Print evaluation report summary"""
        print(report.summary())


if __name__ == "__main__":
    # Example usage
    runner = EvalRunner(data_dir="data", eval_data_dir="eval_data")

    # Generate dataset if needed
    runner.generate_dataset(num_samples=20)

    # Run evaluation
    report = runner.run_evaluation(k_values=[1, 3, 5], run_judge=True)

    # Print and save
    runner.print_report(report)
    runner.save_report(report)

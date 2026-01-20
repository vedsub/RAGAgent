"""
Retrieval Evaluation Metrics

Implements standard IR metrics for evaluating retrieval quality:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- NDCG@K (Normalized Discounted Cumulative Gain)
"""

import math
from typing import List, Set, Dict, Any
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Container for a single metric result"""
    name: str
    value: float
    k: int = None
    
    def __str__(self):
        if self.k:
            return f"{self.name}@{self.k}: {self.value:.4f}"
        return f"{self.name}: {self.value:.4f}"


@dataclass 
class EvalResults:
    """Container for all evaluation results"""
    precision: Dict[int, float]  # k -> precision@k
    recall: Dict[int, float]      # k -> recall@k
    mrr: float
    ndcg: Dict[int, float]        # k -> ndcg@k
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "mrr": self.mrr,
            "ndcg": self.ndcg
        }
    
    def __str__(self):
        lines = ["Evaluation Results:"]
        lines.append(f"  MRR: {self.mrr:.4f}")
        for k in sorted(self.precision.keys()):
            lines.append(f"  Precision@{k}: {self.precision[k]:.4f}")
            lines.append(f"  Recall@{k}: {self.recall[k]:.4f}")
            lines.append(f"  NDCG@{k}: {self.ndcg[k]:.4f}")
        return "\n".join(lines)


class RetrievalMetrics:
    """Compute retrieval evaluation metrics"""
    
    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Compute Precision@K
        
        Precision@K = (# of relevant docs in top K) / K
        
        Args:
            retrieved: Ordered list of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Precision@K score (0 to 1)
        """
        if k <= 0:
            return 0.0
        
        top_k = retrieved[:k]
        relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant)
        
        return relevant_in_top_k / k
    
    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Compute Recall@K
        
        Recall@K = (# of relevant docs in top K) / (total # of relevant docs)
        
        Args:
            retrieved: Ordered list of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Recall@K score (0 to 1)
        """
        if not relevant or k <= 0:
            return 0.0
        
        top_k = retrieved[:k]
        relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant)
        
        return relevant_in_top_k / len(relevant)
    
    @staticmethod
    def mrr(retrieved: List[str], relevant: Set[str]) -> float:
        """
        Compute Mean Reciprocal Rank
        
        MRR = 1 / (rank of first relevant document)
        
        Args:
            retrieved: Ordered list of retrieved document IDs
            relevant: Set of relevant document IDs
            
        Returns:
            MRR score (0 to 1)
        """
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                return 1.0 / rank
        
        return 0.0
    
    @staticmethod
    def dcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Compute Discounted Cumulative Gain at K
        
        DCG@K = sum(rel_i / log2(i + 1)) for i in 1..K
        
        Args:
            retrieved: Ordered list of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            DCG@K score
        """
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k], start=1):
            rel = 1.0 if doc_id in relevant else 0.0
            dcg += rel / math.log2(i + 1)
        
        return dcg
    
    @staticmethod
    def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Compute Normalized Discounted Cumulative Gain at K
        
        NDCG@K = DCG@K / IDCG@K
        
        IDCG is the ideal DCG if all relevant docs were at the top
        
        Args:
            retrieved: Ordered list of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            NDCG@K score (0 to 1)
        """
        # Compute actual DCG
        dcg = RetrievalMetrics.dcg_at_k(retrieved, relevant, k)
        
        # Compute ideal DCG (all relevant docs at top)
        ideal_retrieved = list(relevant)[:k]  # Best case: all relevant first
        idcg = RetrievalMetrics.dcg_at_k(ideal_retrieved, relevant, min(k, len(relevant)))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @classmethod
    def compute_all(
        cls, 
        retrieved: List[str], 
        relevant: Set[str], 
        k_values: List[int] = None
    ) -> EvalResults:
        """
        Compute all metrics for a single query
        
        Args:
            retrieved: Ordered list of retrieved document IDs
            relevant: Set of relevant document IDs
            k_values: List of K values to compute metrics for
            
        Returns:
            EvalResults with all metrics
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]
        
        precision = {}
        recall = {}
        ndcg = {}
        
        for k in k_values:
            precision[k] = cls.precision_at_k(retrieved, relevant, k)
            recall[k] = cls.recall_at_k(retrieved, relevant, k)
            ndcg[k] = cls.ndcg_at_k(retrieved, relevant, k)
        
        mrr_score = cls.mrr(retrieved, relevant)
        
        return EvalResults(
            precision=precision,
            recall=recall,
            mrr=mrr_score,
            ndcg=ndcg
        )
    
    @classmethod
    def aggregate_results(cls, results: List[EvalResults]) -> EvalResults:
        """
        Aggregate results across multiple queries (compute means)
        
        Args:
            results: List of EvalResults from individual queries
            
        Returns:
            EvalResults with averaged metrics
        """
        if not results:
            return EvalResults(precision={}, recall={}, mrr=0.0, ndcg={})
        
        # Get all K values from first result
        k_values = list(results[0].precision.keys())
        
        # Average each metric
        avg_precision = {k: sum(r.precision.get(k, 0) for r in results) / len(results) for k in k_values}
        avg_recall = {k: sum(r.recall.get(k, 0) for r in results) / len(results) for k in k_values}
        avg_ndcg = {k: sum(r.ndcg.get(k, 0) for r in results) / len(results) for k in k_values}
        avg_mrr = sum(r.mrr for r in results) / len(results)
        
        return EvalResults(
            precision=avg_precision,
            recall=avg_recall,
            mrr=avg_mrr,
            ndcg=avg_ndcg
        )


if __name__ == "__main__":
    # Example usage
    retrieved = ["doc_1", "doc_5", "doc_3", "doc_2", "doc_4"]
    relevant = {"doc_3", "doc_2"}
    
    results = RetrievalMetrics.compute_all(retrieved, relevant, k_values=[1, 3, 5])
    print(results)

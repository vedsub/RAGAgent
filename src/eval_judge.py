"""
LLM-as-Judge for Context Relevance Scoring

Uses Groq LLM to evaluate if retrieved context is relevant to the query.
Returns scores on a 1-5 scale with reasoning.
"""

import os
from typing import List, Tuple, Optional
from dataclasses import dataclass
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()


@dataclass
class JudgeScore:
    """Result from LLM judge evaluation"""
    query: str
    context: str
    score: int  # 1-5 scale
    reasoning: str
    
    def __str__(self):
        return f"Score: {self.score}/5 - {self.reasoning[:100]}..."


class LLMJudge:
    """LLM-as-Judge for evaluating context relevance"""
    
    SYSTEM_PROMPT = """You are an expert evaluator assessing the relevance of retrieved context for answering a query.

Your task is to score how relevant the given context is for answering the query on a scale of 1-5:

1 - NOT RELEVANT: The context has no connection to the query
2 - SLIGHTLY RELEVANT: The context mentions related topics but doesn't help answer the query
3 - PARTIALLY RELEVANT: The context contains some useful information but is incomplete
4 - MOSTLY RELEVANT: The context contains most of the information needed to answer the query
5 - HIGHLY RELEVANT: The context directly and completely addresses the query

Be objective and focus on whether the context can help answer the query."""

    EVAL_PROMPT = """Query: {query}

Retrieved Context:
---
{context}
---

Evaluate the relevance of this context for answering the query.

Respond in this exact format:
SCORE: [1-5]
REASONING: [Your brief explanation]"""

    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        """
        Initialize LLM Judge
        
        Args:
            model_name: Groq model to use for judging
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=0.0,  # Deterministic for evaluation
            max_tokens=256,
        )
    
    def _parse_response(self, response: str) -> Tuple[int, str]:
        """Parse score and reasoning from LLM response"""
        lines = response.strip().split("\n")
        
        score = 3  # Default
        reasoning = "Unable to parse response"
        
        for line in lines:
            line = line.strip()
            if line.upper().startswith("SCORE:"):
                try:
                    score_str = line.split(":", 1)[1].strip()
                    # Extract just the number
                    score = int(''.join(c for c in score_str if c.isdigit())[:1])
                    score = max(1, min(5, score))  # Clamp to 1-5
                except (ValueError, IndexError):
                    pass
            elif line.upper().startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()
        
        return score, reasoning
    
    def score_relevance(self, query: str, context: str) -> JudgeScore:
        """
        Score the relevance of context for a query
        
        Args:
            query: The search query
            context: The retrieved context to evaluate
            
        Returns:
            JudgeScore with score (1-5) and reasoning
        """
        prompt = self.EVAL_PROMPT.format(
            query=query,
            context=context[:2000]  # Truncate long contexts
        )
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ])
            
            score, reasoning = self._parse_response(response.content)
            
            return JudgeScore(
                query=query,
                context=context[:500],
                score=score,
                reasoning=reasoning
            )
            
        except Exception as e:
            return JudgeScore(
                query=query,
                context=context[:500],
                score=0,
                reasoning=f"Error during evaluation: {e}"
            )
    
    def batch_score(
        self, 
        query_context_pairs: List[Tuple[str, str]],
        verbose: bool = True
    ) -> List[JudgeScore]:
        """
        Score multiple query-context pairs
        
        Args:
            query_context_pairs: List of (query, context) tuples
            verbose: Print progress
            
        Returns:
            List of JudgeScore results
        """
        results = []
        
        for i, (query, context) in enumerate(query_context_pairs):
            score = self.score_relevance(query, context)
            results.append(score)
            
            if verbose and (i + 1) % 5 == 0:
                print(f"Scored {i + 1}/{len(query_context_pairs)} pairs...")
        
        return results
    
    def compute_average_score(self, scores: List[JudgeScore]) -> float:
        """Compute average relevance score"""
        valid_scores = [s.score for s in scores if s.score > 0]
        if not valid_scores:
            return 0.0
        return sum(valid_scores) / len(valid_scores)
    
    def get_score_distribution(self, scores: List[JudgeScore]) -> dict:
        """Get distribution of scores"""
        distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for score in scores:
            if score.score in distribution:
                distribution[score.score] += 1
        return distribution


if __name__ == "__main__":
    # Example usage
    judge = LLMJudge()
    
    # Test case
    query = "What is the attention mechanism in transformers?"
    context = """The attention mechanism allows the model to focus on different parts 
    of the input sequence when producing output. In transformers, self-attention 
    computes attention scores between all pairs of positions in a sequence, 
    allowing each position to attend to all other positions."""
    
    result = judge.score_relevance(query, context)
    print(f"Query: {query}")
    print(f"Score: {result.score}/5")
    print(f"Reasoning: {result.reasoning}")

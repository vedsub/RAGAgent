#!/usr/bin/env python3
"""
RAG Evaluation CLI

Command-line interface for running RAG evaluations.

Usage:
    # Generate synthetic dataset
    python run_eval.py --generate-dataset --num-samples 50
    
    # Run evaluation with existing dataset
    python run_eval.py --run-eval --k 3 5 10
    
    # Generate dataset AND run evaluation
    python run_eval.py --generate-dataset --num-samples 50 --run-eval --k 3 5 10
    
    # Run evaluation without LLM judge (faster)
    python run_eval.py --run-eval --no-judge --k 3 5
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.eval_runner import EvalRunner


def main():
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Dataset options
    parser.add_argument(
        "--generate-dataset",
        action="store_true",
        help="Generate synthetic evaluation dataset"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples to generate (default: 50)"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to existing dataset JSON file"
    )
    
    # Evaluation options
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Run evaluation"
    )
    parser.add_argument(
        "-k", "--k",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="K values for metrics (default: 1 3 5 10)"
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM judge evaluation (faster)"
    )
    
    # Paths
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing source documents (default: data)"
    )
    parser.add_argument(
        "--store-path",
        type=str,
        default="faiss_store",
        help="Path to FAISS vector store (default: faiss_store)"
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default="eval_data",
        help="Directory for evaluation data/results (default: eval_data)"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename for report (default: auto-generated)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.generate_dataset and not args.run_eval:
        parser.print_help()
        print("\nError: Specify --generate-dataset and/or --run-eval")
        sys.exit(1)
    
    # Initialize runner
    runner = EvalRunner(
        data_dir=args.data_dir,
        eval_data_dir=args.eval_dir
    )
    
    # Load or create vector store
    if not args.quiet:
        print("Loading vector store...")
    runner.load_or_create_vector_store(args.store_path)
    
    # Generate dataset if requested
    if args.generate_dataset:
        if not args.quiet:
            print(f"\nGenerating dataset with {args.num_samples} samples...")
        runner.generate_dataset(num_samples=args.num_samples, save=True)
    
    # Load existing dataset if specified
    if args.dataset_path:
        if not args.quiet:
            print(f"\nLoading dataset from {args.dataset_path}...")
        runner.load_dataset(args.dataset_path)
    
    # Run evaluation if requested
    if args.run_eval:
        if not args.quiet:
            print(f"\nRunning evaluation with k={args.k}...")
        
        report = runner.run_evaluation(
            k_values=args.k,
            run_judge=not args.no_judge,
            verbose=not args.quiet
        )
        
        # Print report
        if not args.quiet:
            print("\n")
        runner.print_report(report)
        
        # Save report
        saved_path = runner.save_report(report, args.output)
        
        if not args.quiet:
            print(f"\nEvaluation complete! Report saved to: {saved_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

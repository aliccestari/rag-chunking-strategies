import argparse
import json
import os
import shutil
from pathlib import Path

from judge_wrapper import *
from run_algorithmic import run_algorithmic_judges

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        dest="input",
        help="Path containing file to run",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        dest="output",
        help="Path to save output",
    )
    parser.add_argument(
        "-e",
        "--algorithmic_evaluators",
        type=str,
        dest="evaluators",
        help="Algorithmic Evaluators configuration file",
        default="scripts/evaluation/config.yaml",
    )
    parser.add_argument(
        "--provider", 
        type=str,
        required=True,
        dest="provider", 
        choices=["openai", "hf", "vllm"],
        help="Provider to use for LLM judges",
    )
    parser.add_argument(
        "--judge_model", 
        type=str, 
        dest="judge_model",
        help="Hugging Face model name (required if provider=hf)"
    )
    parser.add_argument(
        "--openai_key", 
        type=str, 
        help="OpenAI Key (required if provider=openai)"
    )
    parser.add_argument(
        "--azure_host", 
        type=str, 
        help="OpenAI endpoint (required if provider=openai)"
    )
    parser.add_argument(
        "--skip-algorithmic",
        action="store_true",
        dest="skip_algorithmic",
        help=(
            "Do not run DeBERTa/ROUGE/BERTScore/etc. Use -i as JSONL that already has "
            "algorithmic metrics; only run IDK, RAGAS, RadBench, and IDK-conditioned metrics. "
            "If -o differs from -i, -i is copied to -o first."
        ),
    )
    parser.add_argument(
        "--only-idk",
        action="store_true",
        dest="only_idk",
        help="Only run/resume the fast IDK judge. Do not run RAGAS, RadBench, or IDK-conditioned metrics.",
    )
    parser.add_argument(
        "--only-algorithmic",
        action="store_true",
        dest="only_algorithmic",
        help="Only run DeBERTa/ROUGE/BERTScore/etc. and stop before LLM judges.",
    )
    return parser


def jsonl_has_metric(path: str, metric_name: str) -> bool:
    seen = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            seen = True
            metrics = json.loads(line).get("metrics") or {}
            if metric_name not in metrics:
                return False
    return seen

if __name__ == "__main__":
    parser = args_parser()
    args = parser.parse_args()

    if args.skip_algorithmic:
        in_path, out_path = Path(args.input).resolve(), Path(args.output).resolve()
        if in_path != out_path:
            shutil.copyfile(in_path, out_path)
    else:
        run_algorithmic_judges(args.evaluators, args.input, args.output)

    if args.only_algorithmic:
        raise SystemExit(0)
    
    if args.provider == "openai":
        if not args.openai_key or not args.azure_host:
            parser.error("--provider openai requires --openai_key and --azure_host")
            
        os.environ["AZURE_OPENAI_API_KEY"] = args.openai_key
        os.environ["OPENAI_AZURE_HOST"] = args.azure_host
    else:
        if not args.judge_model:
            parser.error(f"--provider {args.provider} requires --judge_model")

        judge_model = args.judge_model
    
    
    if args.provider == "openai":
        if not jsonl_has_metric(args.output, "idk_eval"):
            run_idk_judge(args.provider, "", args.output, args.output)
        if args.only_idk:
            raise SystemExit(0)
        if not jsonl_has_metric(args.output, "RL_F"):
            run_ragas_judges_openai(args.output, args.output, args.openai_key, args.azure_host)
        if not jsonl_has_metric(args.output, "RB_llm"):
            run_radbench_judge(args.provider, "", args.output, args.output)
        
        get_idk_conditioned_metrics(args.output, args.output)
    else:
        if not jsonl_has_metric(args.output, "idk_eval"):
            run_idk_judge(args.provider, args.judge_model, args.output, args.output)
        if args.only_idk:
            raise SystemExit(0)
        
        if not jsonl_has_metric(args.output, "RL_F"):
            run_ragas_judges_local(args.provider, judge_model, args.output, args.output)
        if not jsonl_has_metric(args.output, "RB_llm"):
            run_radbench_judge(args.provider, judge_model, args.output, args.output)
        
        get_idk_conditioned_metrics(args.output, args.output)
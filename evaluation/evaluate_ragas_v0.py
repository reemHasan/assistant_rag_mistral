import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import argparse
import json
import time
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from datasets import Dataset
# ── Judge: LangChain & mistralai for RAGAS ──────
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# ── RAGAS ──────────────────────────────────────────────────────────────────
from ragas import evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    AnswerRelevancy,
    Faithfulness,
    ContextRecall,
    ContextPrecision,
)
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# define metric, and strict generate 1 question for answer relevancy to not supercharge llm
answer_relevancy = AnswerRelevancy(strictness=1)
metrics=[
        Faithfulness(),
        ContextRecall(),
        ContextPrecision(),
        answer_relevancy
    ]

def run_ragas_evaluation(
    testTest_file: str,
    llm_wrapper,
    embed_wrapper,
) -> pd.DataFrame:
    """Run RAGAS metrics, validate results with Pydantic."""
    with open(testTest_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    hf_dataset = Dataset.from_list(dataset)
    #debug_db = hf_dataset.select(range(1))
    logging.info(f"ragas_evaluate on {len(dataset)} test example")
    try:
        result = evaluate(
            dataset    = hf_dataset,
            metrics    = metrics,
            llm        = llm_wrapper,
            embeddings = embed_wrapper,
            run_config = RunConfig(
                max_workers=1,
                timeout=300,
            )
        )
        result_df = result.to_pandas()        
        result_df["category"] = hf_dataset["category"]
        print(result_df.groupby("category")
        .mean(numeric_only=True)
        )
        logging.info("RAGAS evaluation complete")
        return result_df
    except Exception as exc:
        logging.exception(f"RAGAS evaluation error : {exc}")
        raise

# ───────────────────────────────────────────────────────────────────────────
#  Reporting
# ───────────────────────────────────────────────────────────────────────────
def save_outputs(OUTPUT_DIR, results: list[dict], rows: list[dict]) -> None:
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    # CSV — detailed
    df_rows = []
    for r, row in zip(results, rows):
        df_rows.append({
            "id":                  f"{row["category"][:3].upper()}{rows.index(row)+1:02d}",
            "category":            r["category"],
            "question":            r["question"],
            "ground_truth":        row["ground_truth"],
            "answer":              row["answer"],
            "faithfulness":        r["faithfulness"],
            "answer_relevancy":    r["answer_relevancy"],
            "context_recall":      r["context_recall"],
            "context_precision":   r["context_precision"],
            "answer_correctness":  r["answer_correctness"],
            "mean_score":          r["mean_score"],
        })
    df = pd.DataFrame(df_rows)
    csv_path = OUTPUT_DIR / f"ragas_results_{ts}.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")
    # JSON — full detail
    json_path = OUTPUT_DIR / f"ragas_results_{ts}.json"
    json_path.write_text(
        json.dumps([r.model_dump() for r in results], indent=2, ensure_ascii=False)
    )
    print(f"JSON saved:{json_path}")
# ───────────────────────────────────────────────────────────────────────────

def main(modelname, judge_model, judge_embedding):
    """ 
    This function create instance of judge model & embedding
    Load evaluation dataset, pass them to ragas 
    Return ragas results as Pandas.Dataframe
    """
    t0 = time.perf_counter()
    print(" Starting evaluation Rag prototype by RAGAs :")
    llm_wrapper   = LangchainLLMWrapper(judge_model)
    embed_wrapper = LangchainEmbeddingsWrapper(judge_embedding)

    # ── Define path to store evaluation output ──────────────────────────────────────────────────────────
    BASE_DIR = Path(__file__).parent
    eval_dataset = BASE_DIR/"eval_artifacts/eval_dataset_with_answers.json"

    with open(eval_dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print("eval dataset length: ", len(dataset))
    # ── Evaluate ──────────────────────────────────────────────────────────
    print("Step  Running RAGAS evaluation …")
    eval_results = run_ragas_evaluation(eval_dataset, llm_wrapper, embed_wrapper)
    # ── Report ────────────────────────────────────────────────────────────
    csv_path = BASE_DIR/f"eval_artifacts/ragas_{modelname}_results.csv"
    eval_results.to_csv(csv_path, index=False)
    elapsed = time.perf_counter() - t0
    logging.info("Pipeline complete", elapsed_s=round(elapsed, 2))
    print(f"\nDone in {elapsed:.1f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'evaluation pour l'application RAG")
    parser.add_argument(
        "--judge_model",
        type=str,
        default="mistral",
        help="Nom de modele utiliser pour evaluer rag prototype soit mistral ou gemini (par défaut: mistral)"
    )
    args = parser.parse_args()
    print(args.judge_model)
    if args.judge_model == "mistral":
        from utils.config import MISTRAL_API_KEY, EVALUATION_MODEL_MISTRAL, EMBEDDING_MODEL
        # Define llm judge and embedding to be used by Ragas
        judge_llm = ChatMistralAI(
            model   = EVALUATION_MODEL_MISTRAL,
            api_key = MISTRAL_API_KEY,
            temperature = 0,
            )
        judge_embeddings = MistralAIEmbeddings(
            model   = EMBEDDING_MODEL,
            api_key = MISTRAL_API_KEY,
            )
    elif args.judge_model == "gemini":
        from utils.config import GEMINI_API_KEY, EVALUATION_MODEL_NAME, EVALUATION_EMBEDDING
        judge_llm = ChatGoogleGenerativeAI(
            model       = EVALUATION_MODEL_NAME,
            google_api_key = GEMINI_API_KEY,
            temperature = 0,
            n=1,
        )
        judge_embeddings = GoogleGenerativeAIEmbeddings(
            model           = EVALUATION_EMBEDDING,
            google_api_key  = GEMINI_API_KEY,
        )
    else:
        print("Please entre supported model name: mistral or gemini !!! ")
        raise
    main(args.judge_model, judge_llm, judge_embeddings)
"""
Module: run_optimization.py

### About This Module
This module serves as the main experiment execution pipeline for prompt
optimization and evaluation. It coordinates dataset loading,
prompt optimization, performance evaluation, and result storage.

The module supports multiple language models, multiple prompt initialization
strategies, and multiple optimization branches. It automates the complete
experimental workflow required to compare baseline prompts against
iteratively optimized prompts across the sentiment classification and
logical deduction tasks.

### Key Features
For each task and model, the module:
- Downloads and preprocesses the datasets for both tasks.
- Runs prompt optimization procedures using an optimization subset.
- Tracks prompt evolution across generations.
- Evaluates baseline prompts and final otpimized prompts performance on a test subset.
- Records and saves summary metrics and example-level ones in CSV format.
- Saves mutation logs in CSV format for later analysis.

### Main Components
- get_dataset: Downloads benchmark datasets.
- clean_logical: Converts logical deduction data into a unified format.
- clean_sentiment: Converts sentiment data into a unified format.
- LLMModel: Provides local language model inference.
- PromptObject: Represents structured prompts.
- apply_prompt: Builds task-specific prompts.
- evaluate: Computes prompt performance metric.
- MultiBranchOptimizer: Evolves prompts through iterative mutation.

### Functions
- run_experiments: Executes complete optimization experiments.

### Output Files
Results are automatically written to the results directory and include:
- Baseline summary metrics.
- Baseline example predictions.
- Per optimizer configuration optimization summary metrics.
- Per optimizer configuration optimization example predictions.
- Per optimizer configuration prompt mutation logs.
- Per optimizer configuration used prompts.
"""

from models import LLMModel
from evaluator import evaluate
import data_clean
import csv
import init_prompts
from tqdm import tqdm
from MultiBranchOptimizer import MultiBranchOptimizer
import os
import random
import gc

# CSV schema.
ALL_KEYS = [
    "level",
    "prompt",
    "optimizer",
    "generation",
    "task",
    "example_id",
    "pred",
    "truth",
    "exact",
    "judge",
    "optimization_score",
    "test_score",
    "prompt_text",
]


def run_experiments(
    model,
    dataset,
    prompt_dict,
    optimizers,
    task_type,
    judge_model=None,
    max_examples=100,
    generations=50,
    seed=None,
):
    """
    Runs a complete prompt optimization benchmark for a specific task.

    This function applies one or more prompt optimization strategies starting from a
    collection of inital prompts, tracks performance across generations and then
    evaluates the initial prompts and the optimized prompts on a held out test dataset,
    recording their test performance. Results are ontinuously saved to CSV files,
    including aggregate metrics, example-level predictions, and mutation histories.

    The experiment workflow consists of:
    - Running optimization procedures.
    - Recording generation-by-generation performance.
    - Evaluating baseline and optimized prompts.

    Args:
        model: Model used for evaluation.
        dataset (list): Test dataset used for evaluation.
        prompt_dict (dict): Dictionary of initial prompts.
        optimizers (dict of MultiBranchOptimizers): Prompt optimizers.
        task_type (str): Task identifier.
        judge_model (optional): Model used for LLM-based evaluation.
        max_examples (int, optional): Number of examples evaluated per prompt.
        generations (int, optional): Number of optimization generations.
        seed (int, optional): Random seed for reproducibility.
    """

    def save_csv(
        rows,
        path,
        fieldnames,
        extrasaction="raise",
    ):
        """
        Saves rows to a CSV file.

        Args:
            rows (list): Rows to save.
            path (str): Output CSV path.
            fieldnames (list): CSV column names and order.
            extrasaction (str): How to handle unexpected dictionary keys.
        """

        if not rows:
            return

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True,
        )

        with open(
            path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction=extrasaction,
            )

            writer.writeheader()
            writer.writerows(rows)

    if not isinstance(optimizers, dict):
        optimizers = {"optimizer": optimizers}

    rows = []

    # Store prompts that will be evaluated on test set.
    test_candidates = []

    # Save path.
    base_path = f"results/{task_type}/" f"{model.repo_id.split('/')[-1]}"

    total_steps = (
        len(prompt_dict) * len(optimizers) * generations
        + len(prompt_dict)
        + len(prompt_dict) * (1 + len(optimizers))
    )

    pbar = tqdm(
        total=total_steps,
        desc="Running experiments",
    )

    # Loop over initial prompt templates.
    for prompt_name, prompt in prompt_dict.items():

        # Store the baseline prompt for test evaluation.
        test_candidates.append(
            {
                "prompt_name": prompt_name,
                "optimizer": "none",
                "generation": 0,
                "prompt": prompt,
            }
        )

        # Loop over optimizers.
        for opt_name, optimizer in optimizers.items():

            print(f"\nOPTIMIZER: {opt_name}")

            print(f"""
                Running experiment:
                - Task: {task_type}
                - Initial prompt: {prompt_name}
                - Optimizer: {opt_name}
                """)

            # Set current init prompt in log.
            optimizer.current_init_prompt = prompt_name

            # Optimize prompt.
            history, mutation_log = optimizer.optimize(
                initial_prompts=prompt,
                task_type=task_type,
                generations=generations,
            )

            # Store the final optimized prompt for test evaluation.
            optimized_history = [step for step in history if step["gen"] > 0]

            if optimized_history:

                best_final = optimized_history[-1]

                test_candidates.append(
                    {
                        "prompt_name": prompt_name,
                        "optimizer": opt_name,
                        "generation": best_final["gen"],
                        "prompt": best_final["prompt"],
                    }
                )

            # Loop through mutation history.
            for step in history:

                gen = step["gen"]
                opt_prompt = step["prompt"]

                pbar.set_description(
                    f"{task_type} | {prompt_name} | " f"{opt_name} | gen {gen}"
                )

                summary = step["summary"]
                details = step["details"]

                print(
                    f"GEN {gen} OPTIMIZATION SCORE:",
                    summary["score"],
                )

                pbar.update(1)

                # Summary data and evaluation results.
                rows.append(
                    {
                        "level": "summary",
                        "prompt": prompt_name,
                        "optimizer": opt_name,
                        "generation": gen,
                        "task": task_type,
                        "exact": summary["exact"],
                        "judge": summary["judge"],
                        "optimization_score": summary["score"],
                        "test_score": None,
                        "prompt_text": opt_prompt.render(),
                    }
                )

                # Per example data and evaluation results.
                for i, d in enumerate(details):

                    rows.append(
                        {
                            "level": "example",
                            "prompt": prompt_name,
                            "optimizer": opt_name,
                            "generation": gen,
                            "task": task_type,
                            "example_id": i,
                            "pred": d["pred"],
                            "truth": d["truth"],
                            "exact": d["exact"],
                            "judge": d["judge"],
                            "optimization_score": d["score"],
                            "prompt_text": opt_prompt.render(),
                        }
                    )

            # Save mutations log.
            save_csv(
                mutation_log,
                (f"{base_path}/" f"{prompt_name}_{opt_name}_mutations.csv"),
                fieldnames=(mutation_log[0].keys() if mutation_log else []),
            )

            # Save all prompts sent to the mutation model.
            save_csv(
                optimizer.optimizer_prompt_log,
                (f"{base_path}/" f"{prompt_name}_{opt_name}_" f"optimizer_prompts.csv"),
                fieldnames=[
                    "gen",
                    "stage",
                    "branch",
                    "target_field",
                    "optimization_prompt",
                    "optimizer_output",
                ],
                extrasaction="ignore",
            )

    # Test evaluation.

    print("\nTEST EVALUATION")

    for candidate in test_candidates:

        prompt_name = candidate["prompt_name"]
        opt_name = candidate["optimizer"]
        generation = candidate["generation"]
        candidate_prompt = candidate["prompt"]

        print(
            f"\nTEST EVALUATION:"
            f"\n- Prompt: {prompt_name}"
            f"\n- Optimizer: {opt_name}"
            f"\n- Generation: {generation}"
        )

        # Construct test data.
        test_data = init_prompts.apply_prompt(
            dataset,
            candidate_prompt,
            task_type,
        )

        # Evaluate candidate on the test set.
        test_summary, test_details = evaluate(
            model,
            test_data,
            judge_model=judge_model,
            return_details=True,
            max_examples=max_examples,
            seed=seed,
        )

        print(
            "TEST SCORE:",
            test_summary["score"],
        )

        # Baseline.
        if opt_name == "none":

            rows.append(
                {
                    "level": "summary",
                    "prompt": prompt_name,
                    "optimizer": "none",
                    "generation": 0,
                    "task": task_type,
                    "exact": test_summary["exact"],
                    "judge": test_summary["judge"],
                    "optimization_score": None,
                    "test_score": test_summary["score"],
                    "prompt_text": candidate_prompt.render(),
                }
            )

            for i, d in enumerate(test_details):

                rows.append(
                    {
                        "level": "example",
                        "prompt": prompt_name,
                        "optimizer": "none",
                        "generation": 0,
                        "task": task_type,
                        "example_id": i,
                        "pred": d["pred"],
                        "truth": d["truth"],
                        "exact": d["exact"],
                        "judge": d["judge"],
                        "optimization_score": None,
                        "test_score": d["score"],
                        "prompt_text": candidate_prompt.render(),
                    }
                )

        # Final optimized prompt.
        else:

            for row in rows:

                if (
                    row["level"] == "summary"
                    and row["prompt"] == prompt_name
                    and row["optimizer"] == opt_name
                    and row["generation"] == generation
                ):
                    row["test_score"] = test_summary["score"]
                    break

    # Save final results.
    print("\nSAVING TEST RESULTS")

    # Get all baseline results.
    baseline = [r for r in rows if r["optimizer"] == "none"]

    # Save baseline summary results.
    save_csv(
        [r for r in baseline if r["level"] == "summary"],
        f"{base_path}/baseline_summary.csv",
        fieldnames=ALL_KEYS,
    )

    # Save baseline example results.
    save_csv(
        [r for r in baseline if r["level"] == "example"],
        f"{base_path}/baseline_examples.csv",
        fieldnames=ALL_KEYS,
    )

    # Save optimizer results.
    for prompt_name in prompt_dict:

        for opt_name in optimizers:

            subset = [
                r
                for r in rows
                if r["optimizer"] == opt_name and r["prompt"] == prompt_name
            ]

            # Save summary evaluation results.
            save_csv(
                [r for r in subset if r["level"] == "summary"],
                (f"{base_path}/" f"{prompt_name}_{opt_name}_summary.csv"),
                fieldnames=ALL_KEYS,
            )

            # Save per example evaluation results.
            save_csv(
                [r for r in subset if r["level"] == "example"],
                (f"{base_path}/" f"{prompt_name}_{opt_name}_examples.csv"),
                fieldnames=ALL_KEYS,
            )


# Seed for reproducibility.
seed = 24052026
random.seed(seed)

# Get logical deduction dataset for BIG BENCH LITE tasks.
logical_deduction_data = data_clean.get_dataset(
    "https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/logical_deduction/five_objects/task.json"
)

# Clean and prepare logical deduction dataset.
logical_deduction = data_clean.clean_logical(logical_deduction_data["examples"])

# Shuffle logical deduction dataset.
random.shuffle(logical_deduction)

logical_optimization_set = logical_deduction[:300]
logical_test_set = logical_deduction[300:400]

# Get sentiment classification dataset for Natural Instructions tasks.
sentiment_classification_data = data_clean.get_dataset(
    "https://raw.githubusercontent.com/allenai/natural-instructions//master/tasks/task195_sentiment140_classification.json"
)

# Clean and prepare sentiment classification dataset.
sentiment_classification = data_clean.clean_sentiment(
    sentiment_classification_data["Instances"]
)

# Shuffle sentiment classification dataset.
random.shuffle(sentiment_classification)

sentiment_optimization_set = sentiment_classification[:300]
sentiment_test_set = sentiment_classification[300:400]

"""judge = LLMModel(
    repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
    filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    chat_format="llama-3"
)"""

# Models configurations.
model_configs = {
    "qwen": {
        "repo_id": "Qwen/Qwen3-8B-GGUF",
        "filename": "Qwen3-8B-Q4_K_M.gguf",
        "chat_format": None,
    },
    "llama": {
        "repo_id": "m2wws/Llama-3.1-8B-Instruct-Q4_K_M-GGUF",
        "filename": "llama-3.1-8b-instruct-q4_k_m.gguf",
        "chat_format": None,
    },
    "ministral": {
        "repo_id": "DuoNeural/Ministral-8B-Instruct-GGUF",
        "filename": "Ministral-8B-Instruct-Q4_K_M.gguf",
        "chat_format": None,
    },
}


for model_name, config in model_configs.items():

    # Initialize model.
    model = LLMModel(**config)

    print(f"RUNNING MODEL: {model_name}")

    # Initialize optimizers for sentiment classification task.
    optimizers = {
        "rewrite": MultiBranchOptimizer(
            model,
            evaluate,
            sentiment_optimization_set,
            patience=30,
            branches=("rewrite",),
            max_examples=300,
            seed=seed,
        ),
        "instruction": MultiBranchOptimizer(
            model,
            evaluate,
            sentiment_optimization_set,
            patience=30,
            branches=("instruction",),
            max_examples=300,
            seed=seed,
        ),
        "cot": MultiBranchOptimizer(
            model,
            evaluate,
            sentiment_optimization_set,
            patience=30,
            branches=("cot",),
            max_examples=300,
            seed=seed,
        ),
        "examples": MultiBranchOptimizer(
            model,
            evaluate,
            sentiment_optimization_set,
            patience=30,
            branches=("examples",),
            max_examples=300,
            seed=seed,
        ),
        "multi": MultiBranchOptimizer(
            model,
            evaluate,
            sentiment_optimization_set,
            patience=30,
            branches=("instruction", "cot", "examples"),
            max_examples=300,
            seed=seed,
        ),
    }

    # Run optimization experiments for sentiment classification task.
    run_experiments(
        model=model,
        dataset=sentiment_test_set,
        prompt_dict=init_prompts.INITIAL_PROMPTS_SENTIMENT,
        optimizers=optimizers,
        task_type="sentiment",
        judge_model=None,
        max_examples=100,
        seed=seed,
    )

    # Initialize optimizers for logical deduction task.
    optimizers = {
        "rewrite": MultiBranchOptimizer(
            model,
            evaluate,
            logical_optimization_set,
            patience=30,
            branches=("rewrite",),
            max_examples=300,
            seed=seed,
        ),
        "instruction": MultiBranchOptimizer(
            model,
            evaluate,
            logical_optimization_set,
            patience=30,
            branches=("instruction",),
            max_examples=300,
            seed=seed,
        ),
        "cot": MultiBranchOptimizer(
            model,
            evaluate,
            logical_optimization_set,
            patience=30,
            branches=("cot",),
            max_examples=300,
            seed=seed,
        ),
        "examples": MultiBranchOptimizer(
            model,
            evaluate,
            logical_optimization_set,
            patience=30,
            branches=("examples",),
            max_examples=300,
            seed=seed,
        ),
        "multi": MultiBranchOptimizer(
            model,
            evaluate,
            logical_optimization_set,
            patience=30,
            branches=("instruction", "cot", "examples"),
            max_examples=300,
            seed=seed,
        ),
    }

    # Run optimization experiments for logical deduction task.
    run_experiments(
        model=model,
        dataset=logical_test_set,
        prompt_dict=init_prompts.INITIAL_PROMPTS_LOGICAL,
        optimizers=optimizers,
        task_type="logical",
        judge_model=None,
        max_examples=100,
        seed=seed,
    )

    del model
    gc.collect()

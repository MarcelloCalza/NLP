# NLP P8
### Running Experiments and Viewing Results

To run the prompt optimization experiments from scratch, execute run_optimizations.py from the project root directory. The script runs the configured experiments and saves the resulting CSV files in the results/ directory.

The repository already includes the experimental results and analysis plots, so rerunning the experiments is not necessary to inspect the results. The results/ directory contains the saved CSV files, while the analysis/ directory contains the corresponding plots for optimization-score evolution, held-out test performance, and mutation-level improvement rates.

### Results Files

The results folder contains the results files divided by task and then by model:

```text
results/
├── logical/
│   └── <model_name>/
│       ├── baseline_summary.csv
│       ├── baseline_examples.csv
│       ├── <prompt>_<optimizer>_summary.csv
│       ├── <prompt>_<optimizer>_examples.csv
│       ├── <prompt>_<optimizer>_mutations.csv
│       └── <prompt>_<optimizer>_optimizer_prompts.csv
        ...
└── sentiment/
    └── <model_name>/
        └── ...
```

The `<model_name>` directory corresponds to the model repository name used in the experiment. The `<prompt>` identifier specifies the initial prompting strategy (`zero_shot`, `cot`, or `few_shots`), while `<optimizer>` identifies the optimization configuration (`rewrite`, `instruction`, `cot`, `examples`, or `multi`).

| File | Contents |
|---|---|
| `baseline_summary.csv` | Aggregate baseline evaluation results for the initial prompts, including held-out test performance. |
| `baseline_examples.csv` | Per-example baseline predictions, expected answers, and evaluation results. |
| `<prompt>_<optimizer>_summary.csv` | Generation-level optimization history, including the best retained prompt, its optimization score, and the held-out test score where recorded. Generation 0 represents the original prompt; subsequent generations record the optimization history. |
| `<prompt>_<optimizer>_examples.csv` | Per-example evaluation details associated with the prompts recorded in the optimization history. |
| `<prompt>_<optimizer>_mutations.csv` | Records of generated child prompts, including the generation, mutation branch, parent prompt and score, and child prompt and score. Crossover operations record two parent scores. |
| `<prompt>_<optimizer>_optimizer_prompts.csv` | Prompts sent to the mutation model and its outputs, including failure-analysis requests and mutation requests. Records identify the generation, processing stage, branch, and targeted prompt component where applicable. |




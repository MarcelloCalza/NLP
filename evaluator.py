"""
Module: evaluation.py

### About This Module
This module provides utilities for evaluating language model
performance across prompt optimization experiments. It supports
multiple evaluation strategies, including exact matching, keyword
matching, and LLM-based judging.

### Key Features
- Caches model generations to reduce redundant inference calls.
- Computes weighted combinations of evaluation metrics.
- Produces both aggregate and per-example evaluation statistics.

### Functions
- cached_generate: Retrieves cached generations or performs model
  inference when necessary.
- evaluate: Evaluates model predictions using configurable scoring
  metrics and returns aggregate and per-example performance statistics.
"""

import string

cache = {}


def cached_generate(model, prompt, system=None, seed=None):
    """
    Generates a model response while caching previous generations.

    This function checks whether a response has already been generated
    for the specified model, prompt, and system instruction. If a cached
    result exists, it is returned directly. Otherwise, a new response is
    generated, stored in the cache, and returned.

    Args:
        model: Language model wrapper.
        prompt (str): Input prompt.
        system (str, optional): System instruction used to guide model.

    Returns:
        str: Generated model response.
    """

    # Define key.
    key = (
        model.repo_id,
        model.filename,
        prompt,
        system or "",
        seed if seed is not None else "",
    )

    # Generate new response if key isnt in cache.
    if key not in cache:
        cache[key] = model.generate(prompt, system=system, seed=seed)

    return cache[key]


def evaluate(
    model,
    dataset,
    use_exact=False,
    use_keyword=True,
    system=None,
    judge_model=None,
    judge_system=None,
    weights=(0.0, 1.0, 0.0),
    max_examples=None,
    return_details=False,
    seed=None,
):
    """
    Evaluates a language model on a dataset using configurable scoring
    strategies.

    This function generates predictions for each dataset example and
    compares them against the expected outputs. Evaluation can be based
    on exact matching, keyword matching, LLM judging, or a weighted
    combination of these metrics. Individual example scores are
    aggregated into summary statistics describing overall model
    performance.

    Args:
        model: Language model wrapper used for prediction.
        dataset (list): Collection of evaluation examples.
        use_exact (bool, optional): Exact-match accuracy label.
        use_keyword (bool, optional): Keyword-match accuracy label.
        system (str, optional): System instruction provided to the evaluated model.
        judge_model (optional): Language model used for LLM-based judging.
        judge_system (str, optional): System instruction for the judge model.
        weights (tuple, optional): weights assigned to exact match, keyword match,
            and LLM judge scores respectively.
        max_examples (int, optional): Maximum number of examples to evaluate.
        return_details (bool, optional): Per-example evaluation results label.

    Returns:
        dict: Summary evaluation statistics containing:
            - 'exact': Mean exact-match score.
            - 'keyword': Mean keyword-match score.
            - 'judge': Mean LLM-judge score.
            - 'score': Mean weighted score.

        tuple: Returned when return_details is True. Contains:
            - Summary statistics dictionary.
            - List of per-example evaluation results.
    """

    # Normalize/clean output.
    def normalize(text):

        text = text.lower().strip()

        text = text.replace("the ", "")

        text = text.strip(string.punctuation)

        return text

    # Exact match.
    def exact_match(pred, truth):
        return int(normalize(pred) == normalize(truth))

    # Keyword match (check if correct answer is in output).
    def keyword_match(pred, truth, valid_labels):

        pred_norm = normalize(pred)
        tokens = pred_norm.split()

        last_label = None
        last_index = -1

        # Scan token by token.
        for i, tok in enumerate(tokens):
            for label in valid_labels:
                if tok == normalize(label):
                    if i >= last_index:
                        last_index = i
                        last_label = normalize(label)

        # Check if last valid label in prediction is the correct label.
        return int(last_label == normalize(truth))

    # Judge model evaluation.
    def llm_judge(question, pred, truth):

        if judge_model is None:
            return None

        # Prompt for judge model.
        prompt = f"""
                  Question: {question}
                  Model answer: {pred}
                  Correct answer: {truth}
                  Answer ONLY with YES or NO.
                  """

        out = cached_generate(judge_model, prompt, system=judge_system)
        return 1 if normalize(out) == "yes" else 0

    if not (use_exact or use_keyword or judge_model is not None):
        raise ValueError("No evaluation metric enabled")

    results = []

    # Initialize weights.
    w_exact, w_keyword, w_llm = weights

    for i, ex in enumerate(dataset):

        if max_examples and i >= max_examples:
            break

        # Generate model output.
        pred = cached_generate(model, ex["input"], system=system, seed=seed)

        # Extract correct answer.
        truth = ex["output"]

        # Extract labels.
        labels = ex.get("choices", None)
        if labels is None:
            raise ValueError("Missing 'choices' in dataset example")

        # Compute exact match if used.
        em = exact_match(pred, truth) if use_exact else None
        # Compute keyword match if used.
        km = keyword_match(pred, truth, labels) if use_keyword else None
        # Compute judge model evaluation if used.
        jm = llm_judge(ex["input"], pred, truth) if judge_model is not None else None

        # Collect active evaluation methods scores and weights.
        active_scores = []
        active_weights = []
        if use_exact:
            active_scores.append(em)
            active_weights.append(w_exact)
        if use_keyword:
            active_scores.append(km)
            active_weights.append(w_keyword)
        if judge_model is not None:
            active_scores.append(jm)
            active_weights.append(w_llm)

        # Compute weighted score.
        if len(active_weights) > 0:
            weighted_score = sum(
                w * s for w, s in zip(active_weights, active_scores)
            ) / sum(active_weights)
        else:
            weighted_score = 0.0

        # Per example results.
        results.append(
            {
                "pred": pred,
                "truth": truth,
                "exact": em,
                "keyword": km,
                "judge": jm,
                "score": weighted_score,
                "input": ex["input"],
            }
        )

    # Compute mean score.
    def safe_avg(values):
        vals = [v for v in values if v is not None]
        return sum(vals) / len(vals) if vals else None

    # Overall dataset results.
    summary = {
        "exact": safe_avg([r["exact"] for r in results]),
        "keyword": safe_avg([r["keyword"] for r in results]),
        "judge": safe_avg([r["judge"] for r in results]),
        "score": safe_avg([r["score"] for r in results]),
    }

    if return_details:
        return summary, results

    return summary

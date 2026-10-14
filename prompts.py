"""
Module: prompts.py

### About This Module
This module defines the initial prompt templates used throughout the
prompt optimization and evaluation framework.

### Key Features
- Defines baseline zero-shot prompt templates.
- Defines baseline chain-of-thought prompting templates.
- Defines baseline few-shot prompting templates.
- Converts structured datasets into model-ready prompts.

### Objects
- INITIAL_PROMPTS_SENTIMENT:
  Baseline prompt templates for sentiment classification task.

- INITIAL_PROMPTS_LOGICAL:
  Baseline prompt templates for logical reasoning task.

### Functions
- apply_prompt:
  Combines PromptObject templates with dataset examples to generate
  complete prompts suitable for language model inference.
"""

from PromptObject import PromptObject

"""
Initial prompt templates for sentiment classification task.

This dictionary contains the baseline prompt configurations used for
sentiment classification experiments.

The strategies include:
- zero_shot: Task instructions only.
- cot: Task instructions with step-by-step reasoning guidance.
- few_shots: Task instructions with labeled examples.

These prompts serve as starting points for prompt evaluation and
optimization procedures.
"""
INITIAL_PROMPTS_SENTIMENT = {
    "zero_shot": PromptObject(
        instruction="""
Classify the sentiment of the following tweet as positive or negative.
""",
    ),
    "cot": PromptObject(
        instruction="""
Classify the sentiment of the following tweet as positive or negative.
""",
        reasoning="""
1. Identify emotionally relevant words or phrases.
2. Determine whether they express positive or negative sentiment.
3. Decide the overall sentiment.
""",
    ),
    "few_shots": PromptObject(
        instruction="""
Classify the sentiment of the following tweet as positive or negative.
""",
        examples=[
            """
Example 1:
I absolutely loved the movie, it was amazing.
Answer: positive
""",
            """
Example 2:
This is the worst experience I've ever had.
Answer: negative
""",
            """
Example 3:
It was okay, nothing special.
Answer: negative
""",
        ],
    ),
}


"""
Initial prompt templates for logical reasoning tasks.

This dictionary contains the baseline prompt configurations used for
logical deduction experiments.

The available strategies include:
- zero_shot: Task instructions only.
- cot: Task instructions with structured reasoning guidance.
- few_shots: Task instructions with worked examples.

These prompts serve as starting points for prompt evaluation and
optimization procedures.
"""
INITIAL_PROMPTS_LOGICAL = {
    "zero_shot": PromptObject(
        instruction="""
Solve the following logical reasoning problem.
""",
    ),
    "cot": PromptObject(
        instruction="""
Solve the following logical reasoning problem.
""",
        reasoning="""
1. Extract all the constraints.
2. Build the relationships.
3. Check consistency.
4. Identify the correct entity.
""",
    ),
    "few_shots": PromptObject(
        instruction="""
Solve the following logical reasoning problem.
""",
        examples=[
            """
Example 1:
In a museum exhibit, there are five artifacts: statue, painting, vase, mask, and coin.
The statue is older than the painting.
The vase is the newest.
The mask is older than the coin.
The painting is older than the mask.
Which one is the third oldest?
Answer: mask
""",
            """
Example 2:
At a dog show, there are five dogs: bulldog, poodle, beagle, husky, and terrier.
The husky is the heaviest.
The bulldog is heavier than the poodle.
The beagle is lighter than the poodle.
The beagle is heavier than the terrier.
Which one is the lightest?
Answer: terrier
""",
        ],
    ),
}


def apply_prompt(dataset, prompt_obj, task_type):
    """
    Applies a prompt template to a dataset and generates model-ready inputs.

    This function combines task-specific examples with a rendered prompt
    template to produce complete prompts suitable for model inference.

    Sentiment tasks include the tweet text and output constraints,
    whereas logical reasoning tasks include the context, question,
    answer options, and output constraints.

    Args:
        dataset (list | dict): Dataset examples to process.
        prompt_obj (PromptObject): Prompt template.
        task_type (str): Task identifier specifying the prompt format.

    Returns:
        list: List of processed examples.
    """

    new_data = []

    # Render prompt.
    rendered_prompt = prompt_obj.render()

    if isinstance(dataset, dict):
        dataset = [dataset]

    for ex in dataset:

        if task_type == "sentiment":

            # Construct prompt for sentiment classification task.
            full_prompt = (
                rendered_prompt + "\n\n" + f"Tweet: {ex['text']}" + "\n\n" + ex["input"]
            )

        elif task_type == "logical":

            choices = ", ".join(ex["choices"])

            # Construct prompt for logical reasoning task.
            full_prompt = (
                rendered_prompt
                + "\n\n"
                + "Solve:"
                + "\n"
                + ex["context"]
                + "\n"
                + ex["question"]
                + "\n"
                + f"Options: {choices}"
                + "\n\n"
                + ex["input"]
            )

        new_data.append(
            {"input": full_prompt, "output": ex["output"], "choices": ex["choices"]}
        )

    return new_data

from dataclasses import dataclass, field
from typing import List


@dataclass
class PromptObject:
    """
    Represents a structured prompt used during prompt optimization.

    This class stores the different components that may compose a prompt,
    including instructions, reasoning guidance, and examples. It also
    supports a raw prompt mode that bypasses the structured components
    and is returned directly when provided.

    The render method combines the available prompt components into a
    single formatted prompt string that can be sent to a language model.

    Attributes:
        instruction (str): Task instructions provided to the model.
        reasoning (str): Reasoning guidance or chain-of-thought style content.
        examples (List[str]): Collection of examples to include in the prompt.
        raw_prompt (str): Complete prompt text that overrides all structured
            components when provided.
    """

    instruction: str = ""
    reasoning: str = ""
    examples: List[str] = field(default_factory=list)

    raw_prompt: str = ""

    def render(self):
        """
        Renders the prompt as a single formatted string.

        This method combines the available prompt components in a fixed
        order consisting of examples, instructions, and reasoning.

        If a raw prompt is provided, it takes precedence and is returned
        directly without combining the structured components.

        Returns:
            str: Fully rendered prompt ready for model inference.
        """

        if self.raw_prompt.strip():
            return self.raw_prompt.strip()

        blocks = []

        if self.examples:
            blocks.append("\n\n".join(self.examples))

        if self.instruction.strip():
            blocks.append(self.instruction.strip())

        if self.reasoning.strip():
            blocks.append(self.reasoning.strip())

        return "\n\n".join(blocks).strip()

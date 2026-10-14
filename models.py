from llama_cpp import Llama


class LLMModel:
    """
    Wrapper around local Llama language models loaded through llama-cpp.

    This class provides a interface for loading a pretrained
    language model and generating text responses using a chat-based
    prompting format..

    The model is loaded from a Hugging Face repository and configured
    to run with GPU acceleration when available.

    Args:
        repo_id (str): Hugging Face repository containing the model files.
        filename (str): Name of the model file to load.
        chat_format (str): Chat template format used by the model.
    """

    def __init__(self, repo_id, filename, chat_format):
        """
        Initializes and loads a pretrained Llama model.

        This constructor downloads or loads the specified model from a
        Hugging Face repository and configures it for chat-based text
        generation.

        Args:
            repo_id (str): Hugging Face repository containing the model files.
            filename (str): Name of the model file to load.
            chat_format (str): Chat template format used by the model.
        """
        self.repo_id = repo_id
        self.filename = filename
        self.model = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_gpu_layers=-1,
            n_ctx=4096,
            chat_format=chat_format,
            verbose=False,
        )

    def generate(
        self, prompt, system=None, max_new_tokens=300, temperature=0.0, seed=None
    ):
        """
        Generates a response from the language model.

        This method formats the provided prompt as a chat conversation,
        optionally including a system message, and generates a response
        using the underlying Llama model.

        Args:
            prompt (str): User prompt to send to the model.
            system (str, optional): System instruction that guides model behavior.
            max_new_tokens (int, optional): Maximum number of tokens to generate.
            temperature (float, optional): Sampling temperature.

        Returns:
            str: Generated model response.
        """
        messages = []

        # Append system instruction.
        if system is not None:
            messages.append(
                {"role": "system", "content": "/no_think\n" + system}
            )  # "/no_think\n" +

        # Append user prompt.
        messages.append({"role": "user", "content": prompt})

        # Generate model response.
        output = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
        )

        return output["choices"][0]["message"]["content"].strip()

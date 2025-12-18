from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any, Optional
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


PromptType = Union[str, List[Dict[str, str]]]


class ModelInference(ABC):
    """
    Abstract base class for model inference with unified prompt handling.
    """

    def apply_chat_template(self, prompt: PromptType) -> str:
        """
        Convert a prompt (string or chat messages) into a model-ready text string.

        Args:
            prompt (PromptType): Either a single instruction string, or a list of chat messages.

        Returns:
            str: Prompt string to feed into the model.
        """
        if isinstance(prompt, list):
            return self.tokenizer.apply_chat_template(prompt)
        elif isinstance(prompt, str):
            return prompt
        else:
            raise ValueError("Prompt must be either a string or a list of chat messages.")

    @abstractmethod
    def generate(self, prompts: List[PromptType], num_samples: int, **kwargs) -> List[str]:
        """Generate text samples from the given prompt."""
        pass


class HFInference(ModelInference):
    """Hugging Face model inference."""

    def __init__(self, model_name: str, **model_kwargs: Any):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Model loading left as placeholder
        raise NotImplementedError("HFInference model loading is not implemented yet.")

    def generate(self, prompt: List[PromptType], num_sample:int, max_tokens: int = 128, **kwargs) -> List[str]:
        raise NotImplementedError("HFInference generation not implemented.")


class VLLMInference(ModelInference):
    """vLLM model inference wrapper."""

    def __init__(self, model_name: str, sampling_params: SamplingParams, **llm_kwargs: Any):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = LLM(model=model_name, tokenizer=model_name, **llm_kwargs)
        self.sampling_params = sampling_params

    def generate(
        self,
        prompts: List[PromptType],
        num_samples: int = 1,
        **kwargs
    ) -> List[str]:
        self.sampling_params.n = num_samples
        prompts = [self.apply_chat_template(prompt) for prompt in prompts]
        outputs = self.llm.generate(prompts, self.sampling_params)
        results: List[str] = [completion.text for out in outputs for completion in out.outputs]
        return results

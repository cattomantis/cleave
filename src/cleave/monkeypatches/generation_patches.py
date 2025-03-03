import torch
from transformers import (
    GenerationConfig, LogitsProcessorList,
    StoppingCriteriaList, PreTrainedModel,
)
from transformers.generation.streamers import BaseStreamer
from transformers.generation import GenerateDecoderOnlyOutput

from typing import Optional, List, Callable, Union


@torch.inference_mode()
def latent_mode_generate(
    self,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional["PreTrainedModel"] = None,
    streamer: Optional["BaseStreamer"] = None,
    negative_prompt_ids: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[GenerateDecoderOnlyOutput, torch.LongTensor]:
    """This generate method allows us to feed in the last_hidden_state of timestep `t-1`
    to the model to generate the next token at time step `t`.
    
    Note: This is an experimental feature and only works with `GenerationMode GREEDY_SEARCH` and `SAMPLE`.

    Args & Returns:
        Same description for the args as described here:
        https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1906
    
    Example:
    ```python
    
    ```
    """
    ...
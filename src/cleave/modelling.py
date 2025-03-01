import os
import torch
import torch.nn.functional as F
import lightning as L

from transformers import (
    AutoModelForCausalLM,
    Cache, DynamicCache
)
from transformers.utils import logging, LossKwargs
from transformers.processing_utils import Unpack
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from typing import Optional, Union, Tuple


logger = logging.get_logger(__name__)

class CleavedAutoModelForCausalLM(L.LightningModule):
    def __init__(
        self,
        target_model_name_or_path: str | os.PathLike,
        *, cleave_at_index: int,
        adapter_path: os.PathLike = None,
        **kwargs
    ):
        """
        Args:
            target_model_name_or_path (str or os.PathLike): Can be either:
                - a string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                - a path to a *directory* containing model weights saved using the
                  :func:`~transformers.PreTrainedModel.save_pretrained` method, e.g., ``./my_model_directory/``.
                - a path or url to a *saved model archive* (e.g, ``./my_model_directory/model.tar.gz``).
            cleave_at_index (int): Cleave the model at this index. The first `cleave_at_index` layers of the `target_model` are included in the resized `draft_subnet`. The deeper layers (`cleave_at_index+1` onwards) are in the `verify_subnet`.
            adapter_path (os.PathLike, optional): Path to the adapter weights. If provided, the adapter weights will be
                loaded into the model.
            kwargs: Additional keyword arguments passed to the `AutoModelForCausalLM.from_pretrained` method.
        
        Example:
            ```python
            from cleave.modelling import CleavedAutoModelForCausalLM
            cleaved_model = CleavedAutoModelForCausalLM(
                "meta-llama/Llama-3.2-1B-Instruct", cleave_at_index=8
            )
            ```
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = AutoModelForCausalLM.from_pretrained(
            target_model_name_or_path,
            torch_dtype=kwargs.get("torch_dtype") or "auto",
            device_map=kwargs.get("device_map") or "auto",
            **kwargs
        )

        assert cleave_at_index > 0, "cleave_at_index must be positive and non-zero"
        assert cleave_at_index < self.model.config.num_hidden_layers, \
            "cleave_at_index must be less than the number of layers in the model. " \
            f"i.e. {self.model.config.num_hidden_layers}"

        self.cleave_at_index = cleave_at_index

        self.adapter_class = type(self.model.model.layers[-1])

        if adapter_path is not None: ... # To-Do

        else: self.adapter = self.adapter_class(
            config=self.model.config,
            layer_idx=self.cleave_at_index
        )

        # Extracting the aux layers from the model
        self.embed_tokens = self.model.model.embed_tokens
        self.lm_head = self.model.lm_head

    def shallow_subnet_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ):
        """
        Args:
            Refer to 
            https://huggingface.co/docs/transformers/model_doc/llama2#transformers.LlamaForCausalLM
        """
        pass

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Args:
            Refer to 
            https://huggingface.co/docs/transformers/model_doc/llama2#transformers.LlamaForCausalLM
        """
        pass
        

    def training_step(self, batch, batch_idx): ...

    def verify_step(self, batch, batch_idx): ...

    def validation_step(self, batch, batch_idx): ...

    def test_step(self, batch, batch_idx): ...

    def configure_optimizers(self): ...

    def predict_step(self, batch, batch_idx): ...

    def predict(self, *args, **kwargs): ...

    def generate(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def convert_to_hf(self): ...

    def get_shallow_subnet(self): ...

    @property
    def draft_subnet_parameters_count(self):
        return 0 # dummy

    @property
    def verify_subnet_parameters_count(self):
        return 0 # dummy

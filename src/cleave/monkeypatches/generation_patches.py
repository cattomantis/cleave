import os
import torch
import torch.nn as nn
from transformers import (
    Cache, StaticCache,
    GenerationConfig, LogitsProcessorList,
    StoppingCriteriaList, PreTrainedModel,
)
from transformers.utils import (
    ModelOutput,
    is_accelerate_available,
    is_hqq_available,
    is_optimum_quanto_available,
    is_torchdynamo_compiling
)
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GenerationMode, logger
from transformers.generation.utils import (
    GenerateOutput,
    GenerateBeamOutput,
    GenerateNonBeamOutput,
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput
)

import inspect
import warnings
from typing import Optional, List, Callable, Union


def latent_prepare_inputs_for_generation(
    self,
    input_ids: torch.LongTensor,
    past_key_values: Optional[Cache] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """Only difference from the original GenerationMixin.prepare_inputs_for_generation is the fact that we allow for inputs_embeds to be passed to the model even after the first iteration/timestep.

    Args & Returns:
        Same description for the args as described here:
        https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L362

    How to patch:
    ```python
    from cleave.monkeypatches import latent_prepare_inputs_for_generation
    from transformers import AutoModelForCausalLM

    
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # The moneypatched method is now available on the model
    model.prepare_inputs_for_generation = latent_prepare_inputs_for_generation
    ```
    """
    # 1. Handle BC:
    model_inputs = {}
    # - some models don't have `Cache` support (which implies they don't expect `cache_position` in `forward`)
    if self._supports_cache_class:
        model_inputs["cache_position"] = cache_position
    # - `cache_position` was not a mandatory input in `prepare_inputs_for_generation` for those models, and this
    #   function may be called outside of `generate`. Handle most use cases by creating `cache_position` on the fly
    #   (this alternative is not as robust as calling `generate` and letting it create `cache_position`)
    elif cache_position is None:
        past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        cache_position = torch.arange(past_length, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

    # 2. Generic cache-dependent input preparation
    # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
    # Exception 1: when passing input_embeds, input_ids may be missing entries
    # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
    # Exception 3: with synced GPUs cache_position may go out of bounds, but we only want dummy token in that case.
    #              (we can't check exception 3 while compiling)
    # Excpetion 4: If input_embeds are passed then slice it through `cache_position`, to keep only the unprocessed tokens and
    # generate the first token for each sequence. Later use the generated Input ids for continuation.
    if past_key_values is not None:
        model_inputs["past_key_values"] = past_key_values
        if inputs_embeds is not None and input_ids.shape[1] == 0:  # Exception 4
            inputs_embeds = inputs_embeds[:, -cache_position.shape[0] :]
        elif (
            inputs_embeds is not None  # Exception 1
            or (is_torchdynamo_compiling() or cache_position[-1] >= input_ids.shape[1])  # Exception 3
        ):
            input_ids = input_ids[:, -cache_position.shape[0] :]
        elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
            input_ids = input_ids[:, cache_position]

    # 3. Prepare base model inputs
    input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
    # Sush Look here:
    # We are allowing for inputs_embeds to be passed to the model even after the first iteration/timestep.
    if not self.config.is_encoder_decoder:
        if inputs_embeds is not None:
            model_inputs[input_ids_key] = None
            model_inputs["inputs_embeds"] = inputs_embeds
        else:
            # `clone` calls in this function ensure a consistent stride. See #32227
            model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)
            model_inputs["inputs_embeds"] = None
    else:
        model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)

    # 4. Create missing `position_ids` on the fly
    encoder_attention_mask = attention_mask if self.config.is_encoder_decoder else None
    attention_mask = (
        kwargs.pop("decoder_attention_mask", None) if self.config.is_encoder_decoder else attention_mask
    )
    attention_mask_key = "decoder_attention_mask" if self.config.is_encoder_decoder else "attention_mask"
    position_ids_key = "decoder_position_ids" if self.config.is_encoder_decoder else "position_ids"
    if (
        attention_mask is not None
        and kwargs.get(position_ids_key) is None
        and position_ids_key in set(inspect.signature(self.forward).parameters.keys())
    ):
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        kwargs[position_ids_key] = position_ids  # placed in kwargs for further processing (see below)

    # 5. Slice model inputs if it's an input that should have the same length as `input_ids`
    for model_input_name in ["position_ids", "token_type_ids", "decoder_position_ids"]:
        model_input = kwargs.get(model_input_name)
        if model_input is not None:
            if past_key_values is not None:
                current_input_length = (
                    model_inputs["inputs_embeds"].shape[1]
                    if model_inputs.get("inputs_embeds") is not None
                    else model_inputs[input_ids_key].shape[1]
                )
                model_input = model_input[:, -current_input_length:]
                model_input = model_input.clone(memory_format=torch.contiguous_format)
            model_inputs[model_input_name] = model_input

    # 6. Create 4D attention mask is we are using a `StaticCache` (important for performant compiled forward pass)
    if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
        if model_inputs["inputs_embeds"] is not None:
            batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
            device = model_inputs["inputs_embeds"].device
        else:
            batch_size, sequence_length = model_inputs[input_ids_key].shape
            device = model_inputs[input_ids_key].device

        # Create the causal mask with fixed shape in advance, to reduce recompilations. If the function to create
        # the 4D causal mask exists, it should be present in the base model (XXXModel class).
        base_model = getattr(self, self.base_model_prefix, None)
        if base_model is None:
            causal_mask_creation_function = getattr(
                self, "_prepare_4d_causal_attention_mask_with_cache_position", None
            )
        else:
            causal_mask_creation_function = getattr(
                base_model, "_prepare_4d_causal_attention_mask_with_cache_position", None
            )
        if causal_mask_creation_function is None:
            logger.warning_once(
                f"{self.__class__.__name__} has no `_prepare_4d_causal_attention_mask_with_cache_position` method "
                "defined in its base modeling class. Compiled forward passes will be sub-optimal. If you're "
                "writing code, see Llama for an example implementation. If you're a user, please report this "
                "issue on GitHub."
            )
        else:
            attention_mask = causal_mask_creation_function(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.dtype,
                device=device,
                cache_position=cache_position,
                batch_size=batch_size,
                config=self.config,
                past_key_values=past_key_values,
            )
    if attention_mask is not None:
        model_inputs[attention_mask_key] = attention_mask

    if encoder_attention_mask is not None:
        model_inputs["attention_mask"] = encoder_attention_mask

    # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
    for key, value in kwargs.items():
        if key not in model_inputs:
            model_inputs[key] = value

    # 8. Remove unexpected `generate` inputs (TODO @joao: fix trainer and examples)
    model_inputs.pop("labels", None)
    return model_inputs

def latent_sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    """_summary_

    Args & Returns:
        Same description for the args as described here:
        https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L3143

    How to patch:
    ```python
    from cleave.monkeypatches import _latent_sample
    from transformers import AutoModelForCausalLM
    
    
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # The moneypatched method is now available on the model
    model.generate = _latent_sample
    ```
    """
    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

    model_forward = self.__call__
    if isinstance(model_kwargs.get("past_key_values"), Cache):
        is_compileable = model_kwargs["past_key_values"].is_compileable and self._supports_static_cache
        is_compileable = is_compileable and not self.generation_config.disable_compile
        if is_compileable and (
            self.device.type == "cuda" or generation_config.compile_config._compile_all_devices
        ):
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            model_forward = self.get_compiled_call(generation_config.compile_config)

    is_prefill = True
    # To keep track of the last hidden state of the last token
    inputs_embeds_sent_to_model: torch.FloatTensor = None
    while self._has_unfinished_sequences(
        this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
    ):
        # Sush Look here:
        # Only sending the input_ids for the first iteration/timestep.
        # An empty tensor of shape (batch_size, 0) for subsequent iterations/timesteps.
        input_ids_sent_to_model = input_ids if inputs_embeds_sent_to_model is None \
            else torch.tensor([[]] * input_ids.size(0), device=input_ids.device)
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(
            input_ids=input_ids_sent_to_model,
            inputs_embeds=inputs_embeds_sent_to_model,
            **model_kwargs
        )

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        if is_prefill:
            outputs = self(**model_inputs, return_dict=True, output_hidden_states=True)
            is_prefill = False
        else:
            outputs = model_forward(**model_inputs, return_dict=True)

        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue

        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1, :].clone().float()
        next_token_logits = next_token_logits.to(input_ids.device)

        # Sush Look here:
        # We are using the last hidden state of the last token to generate the next token
        next_token_hidden_state = outputs.hidden_states[-1][:, -1, :].clone().float()
        next_token_hidden_state = next_token_hidden_state.to(input_ids.device)
        # Create an alias for readability
        inputs_embeds_sent_to_model = next_token_hidden_state

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids

# Copyright 2024-2025 Daniel Franzen, Jan Disselhoff and David Hartmann
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import torch
import peft
from tokenizers import Tokenizer
from trl import DataCollatorForCompletionOnlyLM


# trl version warning
import trl
assert not trl.__version__.startswith('0.15'), """
WARNING: Do not use this code with trl version 0.15.x!
In combination with unsloth, this will shorten all training inputs
to 1024 tokens, speeding up training, but severely degrading accuracy. 
"""


def _make_causal_mask(input_ids_shape, dtype, device, past_key_values_length=0):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask, dtype, tgt_len=None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class InputMaskingDataCollator(DataCollatorForCompletionOnlyLM):
    def __init__(self, mask_first_n_examples=0, remove_bos_id=None, mask_pad_id=None, stop_token=None,
                 remove_overlaps=True, compress=True, pid_offset=0, replace_last={},
                 attn_dtype=None, full_prompt_attn=False, **kwargs):
        super().__init__(**kwargs)
        self.mask_first_n_examples = mask_first_n_examples
        self.remove_bos_id = remove_bos_id
        self.mask_pad_id = mask_pad_id
        self.stop_token = stop_token
        self.remove_overlaps = remove_overlaps
        self.compress = compress
        self.pid_offset = pid_offset
        self.replace_last = replace_last
        if full_prompt_attn:
            assert attn_dtype is not None, 'required'
        self.attn_dtype = attn_dtype
        self.full_prompt_attn = full_prompt_attn

    def torch_call(self, examples):
        batch = super().torch_call(examples)  # call super, masking all inputs
        b, t = batch['input_ids'].shape
        dev = batch['input_ids'].device

        for i in range(b):
            for _ in range(self.mask_first_n_examples):
                # mask first still unmasked output block
                beg_pos = (batch['labels'][i] != -100).nonzero()
                if not len(beg_pos): break
                beg_pos = beg_pos.min().item()
                end_pos = ((batch['labels'][i][beg_pos:] == -100) & (batch['attention_mask'][i][beg_pos:])).nonzero()
                if not len(end_pos): break
                end_pos = end_pos.min().item() + beg_pos
                batch['labels'][i][beg_pos:end_pos] = -100

        if self.remove_bos_id is not None:
            assert (batch['input_ids'][:, 0] == self.remove_bos_id).all()
            for k in batch:
                batch[k] = batch[k][..., 1:]
            b, t = batch['input_ids'].shape

        # mask pad tokens in inputs and labels
        if self.mask_pad_id is not None:
            batch['attention_mask'][batch['input_ids'] == self.mask_pad_id] = 0
            batch['labels'][batch['labels'] == self.mask_pad_id] = -100
        if self.stop_token is not None:
            for i in range(b):
                last_stop_token = (batch['input_ids'][i] == self.stop_token).nonzero()[-1][0]
                while batch['input_ids'][i, last_stop_token - 1] == self.stop_token:
                    last_stop_token -= 1
                if batch['labels'][i, last_stop_token] != -100:
                    batch['attention_mask'][i, last_stop_token+1:] = 0
                    batch['labels'][i, last_stop_token+1:] = -100

        # remove overlaps
        if self.remove_overlaps:
            batch['labels'][:, 1:][batch['attention_mask'][:, :-1] == 0] = -100

        for k, v in self.replace_last.items():
            for i in range(b):
                last = (batch['input_ids'][i] == k).nonzero()[-1].item()
                batch['input_ids'][i, last] = v

        if (self.compress or self.pid_offset) and 'position_ids' not in batch:
            batch['position_ids'] = torch.arange(t, device=dev)[torch.newaxis].repeat(b, 1) + self.pid_offset

        if self.compress:
            neutral = dict(input_ids=self.tokenizer.pad_token_id, attention_mask=0, labels=-100)
            keep = (batch['attention_mask'] != 0) | (batch['labels'] != -100)
            t = 0
            for i in range(b):
                length = keep[i].sum()
                t = max(t, length)
                for k, v in batch.items():
                    v[i, :length] = v[i][keep[i]]
                    if k in neutral:
                        v[i, length:] = neutral[k]
            if t < batch['input_ids'].shape[-1]:
                for k in batch:
                    batch[k] = batch[k][..., :t]

        if self.attn_dtype is not None and t > 1:
            causal_mask = _make_causal_mask(batch['input_ids'].shape, self.attn_dtype, batch['input_ids'].device)
            if self.full_prompt_attn:
                for i in range(b):
                    beg_last_out = (batch['labels'][i] != -100).nonzero()
                    beg_last_out = beg_last_out[0].item() if len(beg_last_out) else t
                    causal_mask[i, 0, :beg_last_out, :beg_last_out] = 0
            batch['attention_mask'] = _expand_mask(batch['attention_mask'], self.attn_dtype) + causal_mask

        return batch


def load_tf_tokenizer(model_path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path)


def load_tf_model(model_path, bits, dtype=torch.bfloat16, attn_implementation='flash_attention_2', **kw):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    if bits is not None:
        kw['quantization_config'] = {
            8: BitsAndBytesConfig(load_in_8bit=True),
            4: BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=dtype),
        }[bits]

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, attn_implementation=attn_implementation, **kw)
    return model, load_tf_tokenizer(model_path)


def load_unsloth_model(model_name, bits, disable_statistics=True, **kw):
    assert bits in [None, 8, 4]
    if disable_statistics: os.environ['UNSLOTH_DISABLE_STATISTICS'] = '1'
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(model_name, load_in_4bit=bits==4, load_in_8bit=bits==8, **kw)
    if model.max_seq_length == 2048 < model.generation_config.max_length:
        print(f'CHANGING MAX_SEQ_LENGTH {model.max_seq_length} -> {model.generation_config.max_length} (unsloth bug?)')
        to_fix = model
        while to_fix is not None:
            to_fix.max_seq_length = model.generation_config.max_length
            to_fix = getattr(to_fix, 'model', None)
    return model, tokenizer


def save_model_and_tokenizer(store_path, model, tokenizer):
    model.save_pretrained(store_path)
    tokenizer.save_pretrained(store_path)
    to_delete = os.path.join(store_path, 'tokenizer.model')  # delete file, as it interferes with token removal
    if os.path.isfile(to_delete):
        os.remove(to_delete)


def fix_dtypes(model, fix_weights=True, fix_quant_states=True):
    # fix some data types (workaround for unsloth)
    for module in model.modules():
        weight = getattr(module, 'weight', None)
        if weight is not None:
            if torch.is_floating_point(weight):
                if fix_weights and weight.dtype != model.dtype:
                    module.to(model.dtype)
            else:
                qs = getattr(weight, 'quant_state', None)
                if qs is not None:
                    if fix_quant_states and qs.dtype != model.dtype:
                        qs.dtype = model.dtype
    return model


def is_peft_model(model):
    return hasattr(model, 'peft_type')


def merge_peft_into_base(model):
    assert is_peft_model(model)
    return fix_dtypes(model.merge_and_unload())


def get_and_fix_peft_weights(store):
    # change some keys (workaround for added 'modules_to_save')
    state_dict = peft.load_peft_weights(store)
    for k in list(state_dict.keys()):
        if 'modules_to_save' in k:
            del state_dict[k]
            original_module_key = k.replace('.modules_to_save.', '.original_module.')
            if original_module_key in state_dict: del state_dict[original_module_key]
            assert k.replace('.modules_to_save.', '.') in state_dict
    return state_dict


def set_peft_weights(model, state_dict):
    res = peft.set_peft_model_state_dict(model, state_dict)
    assert not res.unexpected_keys, 'error loading weights - some keys not available in model'


def load_peft_state(model, store):
    # convenience method to load peft weights from file and set them for model
    set_peft_weights(model, get_and_fix_peft_weights(store))


def get_or_map_special_tokens(data, mapping=None):
    tokens = set()
    if isinstance(data, dict):
        special = data.get('special_tokens')
        if special is not None:  # find and/or update special token mappings
            for v in special.values():
                tokens.update(v['ids'])
                if mapping is not None:
                    v['ids'] = [mapping.get(i) for i in v['ids'] if i in mapping]
        for v in data.values():  # recursively process dict values
            tokens.update(get_or_map_special_tokens(v, mapping))
    if isinstance(data, list):
        for v in data:  # recursively process lists
            tokens.update(get_or_map_special_tokens(v, mapping))
    return tokens


def remove_tokenizer_normalizer(tokenizer):
    assert tokenizer.is_fast
    tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
    if tokenizer_json.get('normalizer') is not None:
        tokenizer_json['normalizer'] = None
        tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))


def shrink_tokenizer_vocab(tokenizer, keep_indices, keep_special=True, remove_unk=False):
    assert tokenizer.is_fast
    tok_json = json.loads(tokenizer._tokenizer.to_str())
    assert tok_json['model']['type'] == "BPE"

    if keep_special:  # get special tokens to keep
        keep_indices.update(tokenizer.all_special_ids)
        keep_indices.update(get_or_map_special_tokens(tok_json.get('post_processor')))

    if remove_unk:  # remove unknown token
        keep_indices -= {tokenizer.unk_token_id}

    # build mapping from old to new id
    mapping = {old: new for new, old in enumerate(sorted(keep_indices))}

    # update tokenizer info
    tok_json['model']['vocab'] = {k: mapping[v] for k, v in tok_json['model']['vocab'].items() if v in mapping}
    tok_json['model']['merges'] = []
    tok_json['added_tokens'] = [{**t, 'id': mapping[t['id']]} for t in tok_json['added_tokens'] if t['id'] in mapping]
    tok_json['added_tokens'] = sorted(tok_json['added_tokens'], key=lambda t: t['id'])
    get_or_map_special_tokens(tok_json.get('post_processor'), mapping)

    tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tok_json))  # reload json, modifying tokenizer in-place

    if remove_unk:
        tokenizer.unk_token = None

    return mapping  # token mapping to be used later


def shrink_model_embeddings(model, mapping):
    with torch.no_grad():
        # copy embeddings to keep
        row_select = torch.tensor([x[0] for x in sorted(mapping.items(), key=lambda x: x[1])])
        row_select = row_select.to(model.get_input_embeddings().weight.data.device)
        new_embed_t = torch.index_select(model.get_input_embeddings().weight.data, 0, row_select)
        row_select = row_select.to(model.get_output_embeddings().weight.data.device)
        new_lm_head = torch.index_select(model.get_output_embeddings().weight.data, 0, row_select)

        # resize model embeddings
        model.resize_token_embeddings(len(row_select))
        if len(model.get_output_embeddings().weight.data) != len(row_select):  # fix for Llada
            model.get_output_embeddings().weight.data = model.get_input_embeddings().weight.data.clone()
            model.get_output_embeddings().out_features = len(row_select)

        # set to copied values
        model.get_input_embeddings().weight.data[:] = new_embed_t
        model.get_output_embeddings().weight.data[:] = new_lm_head

        # map model tokens to new id
        for config in [model.config, model.generation_config]:
            if config is not None:
                for k, v in list(config.to_dict().items()):
                    if k.endswith('token_id'):
                        setattr(config, k, [mapping.get(t) for t in v] if isinstance(v, list) else mapping.get(v))


def keep_single_char_tokens(model, tokenizer, keep=None, keep_norm=False, keep_model_tok=True, **kwargs):
    if not keep_norm:
        remove_tokenizer_normalizer(tokenizer)  # required for some models
    if keep is None:  # keep all single_length tokens
        keep_indices = set(v for k, v in tokenizer.vocab.items() if len(k) == 1)
    else:  # keep tokens that were passed
        keep_indices = set(tokenizer.vocab[t] for t in keep)
    if keep_model_tok:  # keep tokens used by model
        for config in [model.config, model.generation_config]:
            if config is not None:
                for k, v in config.to_dict().items():
                    if k.endswith('token_id'):
                        keep_indices.update(v if isinstance(v, list) else [v])
    keep_indices -= {None}
    num_embeddings = model.get_input_embeddings().num_embeddings
    keep_indices = {i for i in keep_indices if i < num_embeddings}
    mapping = shrink_tokenizer_vocab(tokenizer, keep_indices, **kwargs)
    shrink_model_embeddings(model, mapping)
    return mapping

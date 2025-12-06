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
import re
import numpy as np
from transformers import get_cosine_schedule_with_warmup
from bitsandbytes.optim import AdamW8bit
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm

from arc_loader import ArcDataset
from model_tools import InputMaskingDataCollator
from model_tools import load_tf_model, keep_single_char_tokens, save_model_and_tokenizer
from model_tools import load_peft_state, merge_peft_into_base
import torch
import torch.nn.functional as F

# get names from script filename
try: model_name = os.path.basename(__file__).split('.')[0].split('_')[1]
except: model_name = None

# input paths
#base_model = 'GSAI-ML/LLaDA-8B-Base'  # auto-downloaded from huggingface.co
base_model = 'pretrained_models/LladaMix1400k-tf-custom-lr45-mt20-keep1-gg2m-cyc85-r512-multigpu-step174960'
arc_data_path = 'input'  # format as on kaggle, auto-downloaded from arc git
re_arc_path = os.path.join('input', 're_arc')  # https://github.com/michaelhodel/re-arc
neoneye_path = os.path.join(arc_data_path, 'arc-dataset-collection', 'dataset')  # https://github.com/neoneye/arc-dataset-collection
sudoku_path = os.path.join(arc_data_path, 'sudoku', 'sudoku-3m.csv')  # https://www.kaggle.com/datasets/radcliffe/3-million-sudoku-puzzles-with-ratings
arc100k_path = os.path.join(arc_data_path, 'arc-gen-100k')  # https://www.kaggle.com/datasets/arcgen100k/the-arc-gen-100k-dataset

# output paths
save_model_path = os.path.join('pretrained_models', model_name or 'interactive_run')

for action in ['train', 'merge']:
    # continue if task already accomplished
    if action == 'train':
        if os.path.exists(f'{save_model_path}-lora'):
            continue
        elif model_name is not None:  # setup logging
            import wandb
            run = wandb.init(project='arc2', name=model_name)

    if action == 'merge' and os.path.exists(f'{save_model_path}-merged'):
        continue

    # load base model & reduce embedding size
    model = tokenizer = None  # free memory
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModel.from_pretrained(base_model, trust_remote_code=True, torch_dtype=torch.bfloat16,
        #quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
    )
    assert getattr(model, 'using_custom_rope_version', None) == 'gg2m', 'please load correct modeling_llada.py'
    import trl; print(f'Trl version: {trl.__version__}')
    #tokenizer.pad_token_id = model.generation_config.pad_token_id = 126084
    #keep_tok = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-=|')+tokenizer.tokenize('\n')+['<|mdm_mask|>']
    #keep_single_char_tokens(model, tokenizer, keep=keep_tok, remove_unk=True)
    mask_token = tokenizer.vocab['<|mdm_mask|>']

    # set formatting options
    fmt_opts = dict(
        preprompt='',
        query_beg='I',
        reply_beg='O',
        reply_end='',
        lines_sep='\n',
        max_tokens=5600,
        borders=dict(x='|', y='-', xy='+'),
        pad_token='.',
        min_size=(32, 31),
        special_tok=['<|mdm_mask|>'],
    )
    newline = tokenizer.vocab[''.join(tokenizer.tokenize(fmt_opts['lines_sep']))]

    # create data collator
    data_collator = InputMaskingDataCollator(
        instruction_template=fmt_opts['query_beg'],
        response_template=fmt_opts['reply_beg'],
        mlm=False,
        tokenizer=tokenizer,
        mask_first_n_examples=0,
        mask_pad_id=tokenizer.vocab[fmt_opts['pad_token']],
        remove_overlaps=False,  # no label shifting for diffusion
        compress=True,
    )
    if data_collator.compress:
        fmt_opts['len_repl'] = [(fmt_opts['pad_token'], ''), (re.compile('<[^<]*>'), 'x')]

    def llada_calculate_loss(model, batch, border_weight=15,
                             I=tokenizer.vocab['I'], O=tokenizer.vocab['O'],
                             B=tokenizer.vocab['-'], R=tokenizer.vocab['|'], X=tokenizer.vocab['+']):
        b, l = batch.input_ids.shape
        dev = batch.input_ids.device
        labels = batch.pop('labels')
        masked_indices = torch.zeros((b, l), device=dev, dtype=bool)
        loss_weight = torch.zeros((b, l), device=dev, dtype=model.dtype)

        for i in range(b):
            start = (batch.input_ids[i] == O).nonzero()[-1, 0] + 1
            mask = batch.input_ids[i, start:] != newline
            masked_indices[i, start:] = mask
            loss_weight[i, start:][mask] = 1
            loss_weight[i, start:][batch.input_ids[i, start:] == B] = border_weight
            loss_weight[i, start:][batch.input_ids[i, start:] == R] = border_weight
            loss_weight[i, start:][batch.input_ids[i, start:] == X] = border_weight
            loss_weight[i, start:] /= b * loss_weight[i, start:].sum()
        batch.input_ids[masked_indices] = mask_token
        logits = model(**batch).logits
        ce_loss = (F.cross_entropy(logits[masked_indices], labels[masked_indices], reduction='none') * loss_weight[masked_indices]).sum()
        with torch.no_grad():
            correct_tokens = (logits[masked_indices].argmax(-1) == labels[masked_indices]).sum().item()
            masked_tokens = masked_indices.sum().item()
        return ce_loss, correct_tokens, masked_tokens

    # create lora model
    lora_layers = ['q_proj', 'k_proj', 'v_proj', 'ff_proj', 'up_proj', 'attn_out', 'ff_out', 'wte']
    model = get_peft_model(model, LoraConfig(
        target_modules=lora_layers,
        r=512,
        lora_alpha=24,
        lora_dropout=0,
        bias="none",
        use_rslora=True,
    ))

    if action == 'train':
        def size_filter(ds, max_known=0.3):
            print('Running dataset filtering...')
            from shape_detection import detect_output_shape
            from collections import defaultdict
            size_unknown = set(k for k, v in ds.challenge.items() if detect_output_shape(**v) is None)
            counter = defaultdict(lambda : [0, 0])
            keys_unknown = []
            for key in ds.keys:
                base_key = key.split('.', 1)[0].split('_', 1)[0]
                name_key = base_key.split('-', 1)[0]
                counter[name_key][base_key in size_unknown] += 1
            keep_frac = {}
            for name_key, (k, u) in counter.items():
                keep_frac[name_key] = min(1, max_known * u / max(1, k))
                print(f'Dataset {name_key}, known={k}, unknown={u}, known_keep_frac={keep_frac[name_key]:.3}')
            new_keys = []
            for key, r in zip(ds.keys, np.random.random((len(ds.keys),))):
                base_key = key.split('.', 1)[0].split('_', 1)[0]
                name_key = base_key.split('-', 1)[0]
                if base_key in size_unknown or r < keep_frac[name_key]: new_keys.append(key)
            count_by_ds = defaultdict(lambda: 0)
            for key in new_keys:
                name_key = key.split('-', 1)[0]
                count_by_ds[name_key] += 1
            print('Examples by dataset:', count_by_ds)
            return ds.change_keys(new_keys)

        # load training data
        arc_v1 = ArcDataset.load_from_contest(year=2024, path=arc_data_path)
        arc_v2 = ArcDataset.load_from_contest(year=2025, path=arc_data_path)
        #arc_gen_100k = ArcDataset.load_from_jsons(os.path.join(arc100k_path, '*'), ex_only=True)
        lim = dict(max_grid_size=30)
        keep_ex = np.array([2, 3, 4, 5, 6])
        train_dataset = ArcDataset.load_from_rearc(re_arc_path, n=644, sizes=keep_ex, seed=50, mix_datasets=dict(
            arc100k=ArcDataset.load_from_jsons(os.path.join(arc100k_path, '*'), ex_only=True).augment(n=224, shfl_keys=True, shfl_ex=True, seed=501, keep_ex=keep_ex+1),
            arc2train=arc_v2['training'].remove_keys(arc_v1['training'].keys).move_test_to_train().augment(n=128, shfl_keys=True, shfl_ex=True, seed=51, keep_ex=keep_ex+1),
            arc1eval=arc_v1['evaluation'].remove_keys(arc_v2['training'].keys).move_test_to_train().augment(n=128, shfl_keys=True, shfl_ex=True, seed=511, keep_ex=keep_ex+1),
            #arc2eval=arc_v2['evaluation'].move_test_to_train().augment(n=128, shfl_keys=True, shfl_ex=True, seed=52, keep_ex=keep_ex+1),
            heavyA=ArcDataset.load_from_jsons(os.path.join(neoneye_path, 'ARC-Heavy', 'data_100k', '*'), **lim).move_test_to_train().augment(n=3, shfl_keys=True, shfl_ex=True, seed=53, keep_ex=keep_ex+1),
            heavyS=ArcDataset.load_from_jsons(os.path.join(neoneye_path, 'ARC-Heavy', 'data_suggestfunction_100k', '*'), **lim).move_test_to_train().augment(n=3, shfl_keys=True, shfl_ex=True, seed=54, keep_ex=keep_ex+1),
            concept=ArcDataset.load_from_neoneye(os.path.join(neoneye_path, 'ConceptARC', 'data', '*', '*'), **lim).move_test_to_train().augment(n=256, shfl_keys=True, shfl_ex=True, seed=55, keep_ex=keep_ex+1),
            #mini=ArcDataset.load_from_neoneye(os.path.join(neoneye_path, 'Mini-ARC', 'data', '*'), **lim).augment(n=32*3, shfl_keys=True, shfl_ex=True, seed=56, keep_ex=keep_ex),
            #pqa=ArcDataset.load_from_neoneye(os.path.join(neoneye_path, 'PQA', 'pqa-dataset', '*', '*'), **lim).max_subtasks(1, seed=57).augment(n=1, shfl_keys=True, shfl_ex=True, seed=57, keep_ex=keep_ex),
            #sudoku=ArcDataset.load_sudoku_csv(sudoku_path, num_challenges=10000, diff_lim=None, examples_per_challenge=6).augment(shfl_keys=True, seed=60, keep_ex=keep_ex),
        )).augment(tp=True, rt=True, perm=True, shfl_ex=True, seed=0)
        train_dataset = size_filter(train_dataset)

        # training config
        batch_size = 1
        grad_acc_steps = 4
        log_each_n_steps = 10
        dataset_text_field = "full"

        # setup optimizer and schedule
        train_steps = len(train_dataset.keys) // (batch_size * grad_acc_steps)
        lora_params = [p for n, p in model.named_parameters() if not ('wte' in n or 'transformer.ff_out' in n)]
        embd_params = [p for n, p in model.named_parameters() if ('wte' in n or 'transformer.ff_out' in n)]
        optimizer = AdamW8bit(
            [{'params': lora_params, 'lr': 5e-5}, {'params': embd_params, 'lr': 5e-6}],
            betas=(0.95, 0.95),
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * train_steps),
            num_training_steps=train_steps,
            num_cycles=0.5,
        )

        # run training
        model.train()
        tokenizer.padding_side = 'right'

        # setup accelerate
        from accelerate import Accelerator
        accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=grad_acc_steps)
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

        model.zero_grad()
        acc_loss = []; acc_corr = acc_mask = 0

        def size_challenge(text, targets=re.compile('[0-9.]'), replace_with=','):
            start = text.rfind('O') + 1
            return text[:start] + targets.sub(replace_with, text[start:])

        with tqdm(range(train_steps * grad_acc_steps), desc='training') as pbar:
            for step in pbar:
                s = step * batch_size
                batch = data_collator([
                    tokenizer(text=size_challenge(train_dataset.get_task(k, len_name=dataset_text_field, **fmt_opts)[1][dataset_text_field]))
                    for k in train_dataset.keys[s: s + batch_size]
                ])
                with accelerator.accumulate(model):
                    loss, correct_tokens, masked_tokens = llada_calculate_loss(model, batch.to(model.device))
                    accelerator.backward(loss)
                    acc_loss.append(loss.item())
                    acc_corr += correct_tokens
                    acc_mask += masked_tokens
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if (step + 1) % (log_each_n_steps * grad_acc_steps) == 0 and acc_loss:
                    mean_loss = sum(acc_loss) / max(1, len(acc_loss))
                    mean_corr = acc_corr / max(1, acc_mask)
                    to_log = {
                        'train/global_step': (step+1) // grad_acc_steps,
                        'train/loss': mean_loss,
                        'train/mean_token_accuracy': mean_corr,
                        'train/learning_rate': optimizer.param_groups[0]['lr'],
                    }
                    pbar.write(str(to_log))
                    try: run.log(to_log)
                    except: print('WANDB LOGGING FAILED')
                    acc_loss = []; acc_corr = acc_mask = 0

        save_model_and_tokenizer(f'{save_model_path}-lora', model, tokenizer)

    if action == 'merge':
        # load peft weights and merge
        load_peft_state(model, f'{save_model_path}-lora')
        model = merge_peft_into_base(model)
        save_model_and_tokenizer(f'{save_model_path}-merged', model, tokenizer)

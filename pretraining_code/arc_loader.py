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
import itertools
import io
import json
import logging
import requests
import zipfile
import hashlib
import numpy as np
from numpy.random import randint
from glob import glob
from tqdm import tqdm
from collections import OrderedDict, defaultdict


SOURCES = {
    2024: 'https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip',
    2025: 'https://github.com/arcprize/ARC-AGI-2/archive/refs/heads/main.zip',
}
FAKE_HASHES = {'a6b7dac3cab03abf2eb333e16610d6dc', '0ae7b8aa16e2de7b2026de0ee0dd91b5'}


def get_from_git(zipfile_url, task_getter=re.compile(r'^ARC-AGI-[^/]*/data/([^/]+)/([^/]+)[.]json')):
    # download zip
    logging.info(f"Downloading arc dataset from '{zipfile_url}'...")
    r = requests.get(zipfile_url)
    assert r.status_code == 200
    z = zipfile.ZipFile(io.BytesIO(r.content))

    # extract subsets
    dataset = defaultdict(dict)
    for f in z.filelist:
        match = task_getter.match(f.filename)
        if match:
            subset, task_id = match.groups()
            dataset[subset][task_id] = json.loads(z.read(f))

    # remove name tags that occur inconsistently in the data
    for subset, tasks in dataset.items():
        for k, v in tasks.items():
            assert v.pop('name', k) == k
        print(f'ARC DOWNLOADER: extracted {len(tasks)} {subset} tasks.')

    return dict(dataset)


class ArcDataset(object):
    def __init__(self, challenge, solutions={}, keys=None, is_fake=False, is_orig=False):
        if keys is None:
            self.keys = []
            for k, v in challenge.items():
                reply_num = len(v['test'])
                self.keys.extend([f'{k}_{i}' for i in range(reply_num)] if reply_num else [k])
            self.keys = sorted(self.keys)
        else:
            self.keys = [k for k in keys]
        base_keys = set(map(self.get_base_key, self.keys))
        self.challenge = {k: challenge[k] for k in base_keys}
        self.solutions = {k: solutions[k] for k in base_keys if k in solutions}
        self.is_fake = is_fake
        self.is_orig = is_orig

    @classmethod
    def load_from_contest(cls, year=None, path=None):
        if not year: assert path, 'either year or path parameter required'
        elif path: path = os.path.join(path, f'arc-prize-{year}')

        # download if required
        if not path or not os.path.exists(path):
            pop_sols = lambda d: {k: [t.pop('output') for t in v['test']] for k, v in d.items()}
            dataset = {k: {'challenge': v, 'solutions': pop_sols(v)} for k, v in get_from_git(SOURCES[year]).items()}
            if path:
                os.makedirs(path, exist_ok=True)
                for name, subset in dataset.items():
                    for part, data in subset.items():
                        with open(os.path.join(path, f"arc-agi_{name}_{part+'s'*(part=='challenge')}.json"), 'w') as f:
                            json.dump(data, f)
            print(f'ARC DOWNLOADER: saved dataset to {path}.')

        # (re)load from path
        if path:
            dataset = defaultdict(dict)
            for match in map(re.compile(r'^arc-agi_([^/_]+)_challenges.json').match, os.listdir(path)):
                if match:
                    subset, = match.groups()
                    with open(os.path.join(path, match.string), 'r') as f:
                        data = f.read()
                        hash = hashlib.md5(data.encode('utf-8')).hexdigest().lower()
                        dataset[subset]['challenge'] = json.loads(data)
                        dataset[subset]['is_fake'] = hash in FAKE_HASHES
                    solutions_file = os.path.join(path, match.string[:-15] + 'solutions.json')
                    if os.path.isfile(solutions_file):
                        with open(solutions_file, 'r') as f:
                            dataset[subset]['solutions'] = json.load(f)

        return {name: cls(**subset, is_orig=True) for name, subset in dataset.items()}


    # loader for Michael Hodel's ReArc https://github.com/neoneye/arc-dataset-collection
    @classmethod
    def load_from_rearc(cls, path, n, sizes, seed, shuffle=True, mix_datasets={},  # loader for ReArc
                        remove_easiest=0, remove_hardest=0, keep_n=None, pre_sort=False, post_sort=False):
        np.random.seed(seed)
        keys = [[] for _ in range(n)]
        challenge = {}
        solutions = {}
        sizes = list(sizes)

        with open(os.path.join(path, 'metadata.json')) as f:
            metadata = json.load(f)

        for i, key in enumerate(tqdm(sorted(metadata.keys()), desc="Data -> load 'rearc'")):
            with open(os.path.join(path, 'tasks', f'{key}.json')) as f:
                tasks = json.load(f)

            tasks = zip(tasks, metadata[key]['pso_difficulties'])
            tasks = sorted(tasks, key=lambda x: x[1])
            total_count = len(tasks)
            tasks = tasks[int(total_count * remove_easiest):]
            tasks = tasks[:total_count - int(total_count * remove_hardest)]
            tasks = np.random.permutation(tasks).tolist()
            if keep_n is not None:
                tasks = tasks[:keep_n]
                assert len(tasks) == keep_n
            if pre_sort:
                tasks = sorted(tasks, key=lambda x: x[1], reverse=True)

            next_sizes = []
            keys_with_diff = []
            for epoch in range(n):
                if not len(next_sizes):
                    next_sizes = np.random.permutation(sizes).tolist()
                next_size_with_test = 1 + next_sizes.pop()
                base_key = f'rearc-{key}{epoch:04x}'
                challenge[base_key] = {'train': [], 'test': []}
                solutions[base_key] = reply = []
                for _ in range(next_size_with_test):
                    if not len(tasks):
                        raise RuntimeError('Not enough examples - generate more re-arc examples or reduce epochs.')
                    example, diff = tasks.pop()
                    challenge[base_key]['train'].append({k: v for k, v in example.items()})
                challenge[base_key]['test'].append(challenge[base_key]['train'].pop())
                solutions[base_key].append(challenge[base_key]['test'][-1].pop('output'))
                keys_with_diff.append((f'{base_key}_0', diff))

            if post_sort:
                keys_with_diff = sorted(keys_with_diff, key=lambda x: x[1])

            for epoch, (subtask_key, diff) in enumerate(keys_with_diff):
                keys[epoch].append(subtask_key)
            #if i>=1: break

        print(f"Data -> 'rearc' size: {sum(map(len, keys))}")
        for name, ds in mix_datasets.items():
            name = cls.base_key_replace_invalid_chars(name)
            print(f"Data -> '{name}' size: {len(ds.keys)}")
            key_map = lambda k: f'{name}-{k}'
            for epoch, ds_keys in enumerate(np.array_split(ds.keys, len(keys))):
                keys[epoch].extend([key_map(k) for k in ds_keys])
            challenge.update({key_map(k): v for k, v in ds.challenge.items()})
            solutions.update({key_map(k): v for k, v in ds.solutions.items()})

        if shuffle:
            keys = [np.random.permutation(epoch).tolist() for epoch in keys]
        keys = [k for epoch in keys for k in epoch]

        print(f"Data -> total size: {len(keys)}")
        return cls(keys=keys, challenge=challenge, solutions=solutions, is_orig=True)

    @classmethod
    def load_from_jsons(cls, pattern, max_grid_size=float('inf'), min_train=1, min_test=1, neoneye_fixes=False, ex_only=False):
        min_test = 0 if ex_only else min_test
        pattern = f'{pattern}.json'
        files = set(glob(pattern))
        if neoneye_fixes:
            for i in itertools.count():
                updated = [fn for fn in files if fn.endswith(f'_v{i + 1}.json')]
                if not updated: break
                for fn in updated:
                    files.remove(fn.replace(f'_v{i + 1}.json', ('.json' if i == 1 else f'_v{i}.json')))
        assert len(files), f"No files found for pattern '{pattern}'."
        challenge = {}
        solutions = {}
        for fn in tqdm(glob(pattern), desc=f"Data -> load '{pattern}'"):
            with open(fn) as f:
                key = cls.base_key_replace_invalid_chars(os.path.split(fn)[-1].replace('.json', ''))
                task = {t: [
                    {i: np.array(a, dtype=int).tolist() for i, a in q.items()}
                    for q in l
                    if all(all(np.array(np.shape(v)) <= max_grid_size) for v in q.values())
                ] for t, l in (dict(train=json.load(f), test=[]) if ex_only else json.load(f)).items()}
                if len(task['train']) >= min_train and len(task['test']) >= min_test:
                    assert key not in challenge, 'duplicate keys'
                    challenge[key] = task
                    solutions[key] = [test_case.pop('output') for test_case in task['test']]
            assert challenge, 'no tasks found'
        return cls(challenge=challenge, solutions=solutions, is_orig=True)

    @classmethod
    def load_from_neoneye(cls, *args, **kwargs):
        return cls.load_from_jsons(*args, neoneye_fixes=True, **kwargs)

    @classmethod
    def load_sudoku_csv(cls, path, examples_per_challenge=3, num_challenges=10, diff_lim=None, diff_lim_lower=None, min_indx=None, max_indx=None):
        import pandas as pd
        df = pd.read_csv(path)

        if max_indx is not None:
            df = df[:max_indx]

        if min_indx is not None:
            df = df[min_indx:]

        if diff_lim is not None:
            df = df[df['difficulty'] < diff_lim]

        if diff_lim_lower is not None:
            df = df[df['difficulty'] > diff_lim_lower]

        print(f"Loaded {len(df)} sudoku puzzles.")

        df['puzzle'] = df['puzzle'].apply(lambda x: x.replace(".", "0"))

        challenges = {}
        solutions = {}
        difficulty = {}

        for j in range(num_challenges):
            examples = []
            for i in range(j * (examples_per_challenge + 1), (j + 1) * (examples_per_challenge + 1)):
                row = df.iloc[i]
                puzzle = np.array([int(x) for x in row['puzzle']]).reshape(9, 9)
                solution = np.array([int(x) for x in row['solution']]).reshape(9, 9)
                examples.append({"input": puzzle, 'output': solution})

            id = f"sudoku{df.iloc[i]['id']}"
            last = examples.pop()
            solutions[id] = [last.pop('output')]
            challenges[id] = {'train': examples, 'test': [last]}
            difficulty[id] = row['difficulty']

        return_class = cls(challenge=challenges, solutions=solutions, is_orig=True)
        return_class.difficulty = difficulty

        return return_class

    def change_keys(self, keys, is_orig=False):
        return self.__class__(challenge=self.challenge, solutions=self.solutions, keys=keys, is_orig=is_orig)

    def remove_keys(self, keys):
        keys = set(keys)
        return self.change_keys([k for k in self.keys if k not in keys], is_orig=self.is_orig)

    def max_subtasks(self, n, seed=None):
        assert self.is_orig
        if seed is not None:
            np.random.seed(seed)
        new_challenge = {}
        new_solutions = {}
        for k in self.challenge.keys():
            subset = (np.arange if seed is None else np.random.permutation)(len(self.challenge[k]['test']))
            if seed is not None:
                subset = np.random.permutation(subset)
            subset = subset[:n]
            new_challenge[k] = {
                'train': self.challenge[k]['train'],
                'test': [self.challenge[k]['test'][i] for i in subset],
            }
            if k in self.solutions:
                new_solutions[k] = [self.solutions[k][i] for i in subset]
        return self.__class__(challenge=new_challenge, solutions=new_solutions, is_orig=True)

    def split(self, n, split_seed, **kwargs):
        assert self.is_orig, 'Must be run on original dataset.'
        keys = sorted(self.challenge.keys())
        if split_seed == 'len':
            keys = self.sort_keys_by_len(keys=keys, **kwargs)
        else:
            assert isinstance(split_seed, int)
            assert not kwargs
            np.random.seed(split_seed)
            keys = np.random.permutation(keys)
        split_datasets = []
        for new_keys in np.array_split(keys, n):
            new_challenge = {k: self.challenge[k] for k in new_keys}
            split_datasets.append(self.__class__(challenge=new_challenge, solutions=self.solutions, is_orig=True))
        return split_datasets

    def split_off_keys(self, keys):
        assert self.is_orig, 'Must be run on original dataset.'
        new_challenge = {k: self.challenge[k] for k in keys}
        return self.__class__(challenge=new_challenge, solutions=self.solutions, is_orig=True)

    def remove_test_data(self):
        assert self.is_orig, 'Must be run on original dataset.'
        new_challenge = {k: {'train': v['train'], 'test': []} for k, v in self.challenge.items()}
        return self.__class__(challenge=new_challenge)

    def remove_solutions(self):
        return self.__class__(keys=self.keys, challenge=self.challenge, is_orig=self.is_orig)

    def prepare_for_ttt(self):
        assert not self.solutions, 'Challenges must be empty.'
        new_keys = []
        new_challenge = {}
        new_solutions = {}
        for k in self.keys:
            assert k == k.split('_')[0]
            assert not self.challenge[k]['test']
            n = len(self.challenge[k]['train'])
            for i in range(n):
                if k not in new_challenge:
                    new_challenge[k] = {n: [ex.copy() for ex in self.challenge[k]['train']] for n in ['train', 'test']}
                    new_solutions[k] = [ex.pop('output') for ex in new_challenge[k]['test']]
                examples = list(range(n))
                del examples[i]
                new_keys.append(self.permute_ex(f'{k}_{i}', perm=examples))
        return self.__class__(keys=new_keys, challenge=new_challenge, solutions=new_solutions)

    @staticmethod
    def base_key_replace_invalid_chars(base_key):
        return base_key.replace('_', '-').replace('.', '-')

    @staticmethod
    def get_base_key_and_reply_num(key):
        key_num = key.split('.', 1)[0]
        base_key, reply_num = key_num.split('_') if '_' in key_num else (key_num, -1)
        return base_key, int(reply_num)

    @classmethod
    def get_base_key(cls, key):
        return cls.get_base_key_and_reply_num(key)[0]

    def grouped_keys(self):
        grouped_keys = OrderedDict()
        for key in self.keys:
            base_key, reply_num = self.get_base_key_and_reply_num(key)
            if base_key not in grouped_keys:
                grouped_keys[base_key] = []
            while len(grouped_keys[base_key])<=reply_num:
                grouped_keys[base_key].append([])
            grouped_keys[base_key][reply_num].append(key)
        return grouped_keys

    def move_test_to_train(self):
        assert self.is_orig, 'Must be run on original dataset.'
        new_challenge = {}
        for k, v in self.challenge.items():
            new_challenge[k] = {
                'train': v['train'] + [{**t, 'output': self.solutions[k][i]} for i, t in enumerate(v['test'])],
                'test': []
            }
        return self.__class__(challenge=new_challenge, is_orig=self.is_orig)

    @staticmethod
    def permute_array(a, descriptor, invert=False):
        permutation = [int(i) for i in descriptor if str(i).isdigit()]
        assert sorted(permutation) == list(range(10))
        a = np.asarray(a)
        if (a.ndim == 2) == bool(invert): permutation = np.argsort(permutation)
        if (a.ndim == 2):
            permutation = np.concatenate([permutation, np.arange(10, a.max()+1)])
            return np.asarray(permutation)[a]
        assert a.ndim == 3
        return a[..., permutation]
        
    @classmethod
    def transform_array(cls, array, transforms, apply_perm=True, invert=False):
        if array is None: return None
        array = np.asarray(array)
        if invert: transforms = transforms[::-1]
        for tf in transforms:
            if tf == 'tp':
                array = np.swapaxes(array, 0, 1)
            if tf == 'rt':
                array = np.rot90(np.rot90(np.rot90(array)) if invert else array)
            if apply_perm and tf.startswith('perm'):
                array = cls.permute_array(array, tf, invert=invert)
        return array

    @classmethod
    def fmt_array(cls, array, lines_sep, special_tok=[], tf=None, borders={}, min_size=None, pad_token=None):
        if tf is not None:
            array = cls.transform_array(array, tf)
        lines = [[str(c if c<10 else special_tok[c-10]) for c in row] + [borders.get('x', '')] for row in array]
        if 'y' in borders or 'yx' in borders:
            lines.append([])
            if 'y' in borders: lines[-1].extend([borders['y']]*array.shape[-1])
            if 'xy' in borders: lines[-1].append(borders['xy'])
        if min_size is not None:
            min_y, min_x = min_size
            lines.extend([[]] * (min_y - len(lines)))
            for line in lines:
                line.extend([pad_token]*(min_x - len(line)))
        return lines_sep.join(''.join(line) for line in lines)

    @classmethod
    def fmt_input(cls, array, query_beg, reply_beg, **kwargs):
        return query_beg + cls.fmt_array(array, **kwargs) + reply_beg

    @classmethod
    def fmt_output(cls, array, reply_end, **kwargs):
        return cls.fmt_array(array, **kwargs) + reply_end

    @classmethod
    def fmt_train(cls, train_ex, preprompt, query_beg, reply_beg, reply_end, **kwargs):
        examples = [cls.fmt_input(x['input'], query_beg, reply_beg, **kwargs) +
                    cls.fmt_output(x['output'], reply_end, **kwargs) for x in train_ex]
        return preprompt + ''.join(examples)

    def fmt_task(self, key, preprompt, query_beg, reply_beg, reply_end, query_only=False, reply=True, challenge_pos=None, **kwargs):
        key_num, *tf = key.split('.')
        base_key, reply_num = self.get_base_key_and_reply_num(key_num)
        data_train = self.challenge[base_key]['train']
        data_query = self.challenge[base_key]['test']
        if reply is True:
            reply = self.solutions[base_key][reply_num] if base_key in self.solutions and reply_num >= 0 else None
        elif reply is not None:
            assert reply_num >= 0
        for t in tf:
            if t.startswith('ex'):
                data_train = [data_train[int(i)] for i in t[2:].split('-')] if len(t[2:]) else []
        ret = dict(key=key)
        ret['train'] = self.fmt_train(data_train, preprompt, query_beg, reply_beg, reply_end, tf=tf, **kwargs)
        ret['query'] = self.fmt_input(data_query[reply_num]['input'], query_beg, reply_beg, tf=tf, **kwargs) if reply_num >= 0 else ''
        ret['input'] = ret['train'] + ret['query'] if reply_num >= 0 else ''
        if reply is not None:
            ret['reply'] = self.fmt_output(reply, reply_end, tf=tf, **kwargs)
        if challenge_pos is None:
            ret['full'] = ret['train'] + (ret['query'] + ('' if query_only else ret['reply']) if reply is not None else '')
        else:
            assert reply is not None
            challenge_pos = challenge_pos % (len(data_train) + 1)
            beg = self.fmt_train(data_train[:challenge_pos], preprompt, query_beg, reply_beg, reply_end, tf=tf, **kwargs)
            end = self.fmt_train(data_train[challenge_pos:], '', query_beg, reply_beg, reply_end, tf=tf, **kwargs)
            ret['full'] = beg + (ret['query'] + ('' if query_only else ret['reply'])) + end
        return ret

    def get_task(self, key, max_tokens=None, len_name=None, len_repl=None, **kwargs):
        while True:
            fmt = self.fmt_task(key, **kwargs)
            if max_tokens is None or self.count_tokens(fmt[len_name], len_repl) <= max_tokens:
                break
            if not key.split('.')[-1].startswith('ex'):
                base_key = self.get_base_key(key)
                key = f"{key}.ex{'-'.join(map(str, range(len(self.challenge[base_key]['train']))))}"
            key_split = key.split('.')
            key_split[-1] = '-'.join(key_split[-1].split('-')[:-1])
            assert len(key_split[-1]) > 2 and key_split[-1].startswith('ex')
            key = '.'.join(key_split)
        return key, fmt

    @staticmethod
    def count_tokens(data, len_repl=None):
        for k, v in ([(re.compile('<[^<]*>'), 'x')] if len_repl is None else len_repl):
            data = data.replace(k, v) if isinstance(k, str) else k.sub(v, data)
        return len(data)

    @classmethod
    def max_new_tokens(cls, max_size=30, safety_margin=1, len_repl=None, **kwargs):
        max_sized_reply = np.zeros([max_size, max_size], dtype=int)
        fwd = ['reply_end', 'lines_sep', 'borders', 'min_size', 'pad_token']
        fmt = cls.fmt_output(max_sized_reply, **{k: v for k, v in kwargs.items() if k in fwd})
        return cls.count_tokens(fmt, len_repl) + safety_margin

    def get_length(self, key, len_name, max_of_transposed=False, len_repl=None, max_tokens='unused', **fmt_opts):
        if not fmt_opts:
            fmt_opts = dict(preprompt='', query_beg='', reply_beg='', reply_end='', lines_sep='')
            length = self.count_tokens(self.fmt_task(key, **fmt_opts)[len_name], len_repl)
        else:
            length = self.count_tokens(self.fmt_task(key, **fmt_opts)[len_name], len_repl)
            if max_of_transposed:
                length = max(length, self.count_tokens(self.fmt_task(f'{key}.tp', fmt_opts)[len_name], len_repl))
            length += 1  # for bos token
        return length

    def sort_keys_by_len(self, keys, reverse=False, **kwargs):
        lengths = [(key, self.get_length(key, **kwargs)) for key in keys]
        return [x[0] for x in sorted(lengths, reverse=reverse, key=lambda x: x[1])]

    def sorted_by_len(self,**kwargs):
        return self.change_keys(self.sort_keys_by_len(self.keys, **kwargs))

    def convert_with_token_limit(self, quiet=False, **kwargs):
        out_list = []
        new_keys = []
        for key in tqdm(self.keys, desc='convert dataset', disable=quiet):
            key, fmt = self.get_task(key, **kwargs)
            new_keys.append(key)
            out_list.append(fmt)
        return out_list, self.change_keys(new_keys)

    def as_list(self, **kwargs):
        return self.convert_with_token_limit(**kwargs)[0]

    @staticmethod
    def rand_perm(n, sep=None, keep_zero=False):
        permutation = np.random.permutation(n).tolist()
        if keep_zero:
            permutation = [0] + [x for x in permutation if x != 0]
        return permutation if sep is None else sep.join(map(str, permutation))

    @staticmethod
    def permute_ex(key, perm, keep_max=None):
        split = key.split('.')
        for i in range(1, len(split)):
            if split[i].startswith('ex'):
                current_ex = split.pop(i)[2:]
                current_ex = [int(i) for i in current_ex.split('-')] if current_ex else []
                perm = [current_ex[i] for i in perm if i<len(current_ex)] if perm is not None else current_ex
                break
        if keep_max is not None: perm = perm[:keep_max]
        if perm is not None: split.append(f"ex{'-'.join(map(str, perm))}")

        return '.'.join(split)

    def augment_keys(self, keys, tp=False, rt=False, n=1, shfl_keys=False, perm=False, keep_bg=False, shfl_ex=False, keep_ex=None):
        passes = int(n - 1e-4) + 1
        if tp == 'all': keys = [k + n * '.tp' for n in range(2) for k in keys]
        if rt == 'all': keys = [k + n * '.rt' for n in range(4) for k in keys]
        if isinstance(keep_ex, (list, tuple, np.ndarray)):
            def keep_max(next_ex=[]):
                if not next_ex: next_ex.extend(np.random.permutation(keep_ex).tolist())
                return next_ex.pop()
        else: keep_max = lambda: keep_ex
        ex = lambda k: len(self.challenge[self.get_base_key(k)]['train'])
        augmented = [[self.permute_ex(key=k + bool(tp and tp != 'all') * randint(0, 2) * '.tp'
                                            + bool(rt and rt != 'all') * randint(0, 4) * '.rt'
                                            + bool(perm) * ('.perm' + self.rand_perm(10, '', keep_bg)),
                    perm=[None, [np.arange(ex(k)), self.rand_perm(ex(k))][bool(shfl_ex)]][bool(shfl_ex or keep_ex is not None)],
                    keep_max=keep_max(),
                    ) for k in [keys, np.random.permutation(keys)][bool(shfl_keys)]] for _ in range(passes)]
        if passes - n > 1e-8:
            augmented[0] = augmented[0][:max(0, int(len(augmented[0]) * (n + 1 - passes + 1e-8)))]
        return [k for epoch in augmented for k in epoch]

    def augment(self, seed, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        return self.change_keys(self.augment_keys(self.keys, **kwargs))

    def decode(self, text, lines_sep, key=None):
        correct, info = None, 'unknown'
        try:
            data = [[int(x) for x in row if x.isdigit()] for row in text.split(lines_sep)]
            data = [row for row in data if len(row)]
            data = np.array(data, dtype=int)
            assert data.ndim == 2 and all(0 < x <= 30 for x in data.shape)
        except:
            data = None
            correct, info = False, 'cant_decode'
        if key is not None and data is not None:
            key_num, *transforms = key.split('.')
            base_key, reply_num = self.get_base_key_and_reply_num(key_num)
            data = self.transform_array(data, transforms, invert=True)
            correct_solution = self.solutions.get(base_key)
            if correct_solution is None:
                info = 'sol_unknown'
            else:
                correct_solution = np.asarray(correct_solution[reply_num])
                if np.array_equal(correct_solution, data):
                    correct, info = True, 'ALL_CORRECT'
                else:
                    correct, info = False, ('bad_content' if correct_solution.shape == data.shape else 'bad_xy_size')
        return data, correct, info

    def get_submission(self, results=None):
        assert self.is_orig, 'Must be run on original dataset.'
        submission = {k: [{f'attempt_{i+1}': [[0]] for i in range(2)} for _ in range(len(v['test']))] for k, v in self.challenge.items()}
        if results is not None:
            self.fill_submission(results, submission)
        return submission

    @staticmethod
    def fill_submission(results, submission):
        for base_key, data in results.items():
            for reply_num, guesses in enumerate(data):
                target_dict = submission[base_key][reply_num]
                for i, g in enumerate(guesses[:len(target_dict)]):
                    target_dict[f'attempt_{i + 1}'] = g['output'].tolist()

    def validate_submission(self, submission):
        assert self.is_orig, 'Must be run on original dataset.'
        assert self.solutions, 'Solutions must be loaded for submission verification.'
        score = 0
        for k, v in self.solutions.items():
            for i, r in enumerate(v):
                for attempt in ['attempt_1', 'attempt_2']:
                    if np.array_equal(r, submission[k][i][attempt]):
                        score += 1 / len(v)
                        break
        return score

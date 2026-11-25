"""Extract exact pinned upstream functions without entrypoint imports."""
import ast
import json
from pathlib import Path
from copy import deepcopy
from typing import List

HERE = Path(__file__).resolve().parent

def provenance():
    """provenance/REPL.json: beside this folder in the Modal image, else at the repository root."""
    for folder in (HERE, *HERE.parents):
        path = folder / 'provenance' / 'REPL.json'
        if path.is_file():
            return json.loads(path.read_text())
    raise FileNotFoundError('provenance/REPL.json not found above ' + str(HERE))

PROVENANCE = provenance()
REVISION = PROVENANCE['upstream']['revision']
# Download-marker revisions of the cached Hub snapshots under /models/<name>.
ASSETS = {'base': PROVENANCE['base_model']['revision'],
          'adapter-synth-docs': PROVENANCE['adapter_synth_docs']['revision'],
          'tokenizer': PROVENANCE['tokenizer']['revision']}

def source_functions(filename, names, namespace):
    path = HERE / 'vendor' / filename
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace

def expansion_function():
    return source_functions('sft.py', ['apply_chat_template'], {
        'deepcopy': deepcopy, 'List': List, 'PreTrainedTokenizerBase': object,
    })['apply_chat_template']

def audit_masks(dataset, collator, batch_size=4):
    """Check real TRL masks against labels; reject zero-loss rows. Integer lengths."""
    rows = []
    for offset in range(0, len(dataset), batch_size):
        samples = [dataset[i] for i in range(offset, min(offset + batch_size, len(dataset)))]
        batch = collator(samples)
        for j, sample in enumerate(samples):
            ids, mask = sample['input_ids'], sample['completion_mask']
            labels = batch['labels'][j].tolist()
            assert len(ids) == len(mask) <= 2048
            assert all((labels[k] != -100) == bool(mask[k]) for k in range(len(ids)))
            assert all(x == -100 for x in labels[len(ids):])
            trained = sum(x != -100 for x in labels)
            assert trained > 0, 'Completion fully truncated: cannot profile zero-loss row'
            rows.append({'index': offset + j, 'total_tokens': int(len(ids)),
                         'completion_tokens': int(trained), 'prompt_tokens': int(len(ids)-trained)})
    return rows


def device_placement(model):
    """Optional diagnostics: single-device Trainer may remove hf_device_map."""
    mapping = getattr(model, 'hf_device_map', None)
    return {'device_map': {k: str(v) for k, v in mapping.items()} if isinstance(mapping, dict) else {},
            'parameter_devices': sorted({str(parameter.device) for parameter in model.parameters()})}

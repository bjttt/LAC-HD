"""Read the paper's fixed dialogue-group split; never generate a two-way split."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

DEFAULT_SPLIT_MANIFEST = str(Path(__file__).resolve().parents[2] / 'LAC-HD_data/splits/halueval_dialogue_group_90_5_5_seed42.json')
MANIFEST_NAME = 'halueval_dialogue_group_90_5_5_seed42'

def dialogue_group_id(row):
    match = re.fullmatch(r'(dialogue_\d+)_(right|hallucinated)', str(row.get('id', '')))
    if match is None:
        raise ValueError('Expected a retained HaluEval dialogue response ID')
    return match.group(1)

def fixed_split_indices(rows, manifest_path=DEFAULT_SPLIT_MANIFEST):
    raw = Path(manifest_path).read_bytes()
    manifest = json.loads(raw.decode('utf-8-sig'))
    parts = manifest['partitions']
    if manifest.get('name') != MANIFEST_NAME or manifest.get('seed') != 42 or set(parts) != {'train', 'dev', 'test'}:
        raise ValueError('Use the fixed paper 90/5/5 seed-42 train/dev/test manifest')
    all_ids = [str(row['id']) for row in rows]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError('Dataset contains duplicate IDs')
    groups_by_id = {str(row['id']): dialogue_group_id(row) for row in rows}
    seen_ids, seen_groups = set(), set()
    indices = {}
    expected = {'train': (9000, 17847), 'dev': (500, 991), 'test': (500, 991)}
    for name in ('train', 'dev', 'test'):
        part = parts[name]
        ids, groups = set(part['sample_ids']), set(part['group_ids'])
        if len(ids) != len(part['sample_ids']) or len(groups) != len(part['group_ids']):
            raise ValueError('Duplicate split membership')
        if (len(groups), len(ids)) != expected[name]:
            raise ValueError(f'Unexpected paper partition counts: {name}')
        if ids & seen_ids or groups & seen_groups:
            raise ValueError('Overlapping split IDs or dialogue groups')
        if {sid for sid, group in groups_by_id.items() if group in groups} != ids:
            raise ValueError(f'Dataset IDs/groups differ from the frozen manifest: {name}')
        seen_ids.update(ids)
        seen_groups.update(groups)
        indices[name] = [i for i, sid in enumerate(all_ids) if sid in ids]
    if seen_ids != set(all_ids):
        raise ValueError('Dataset and manifest must cover exactly the same retained responses')
    info = {'protocol': 'frozen_90_5_5', 'split_mode': 'manifest',
            'manifest': str(Path(manifest_path).resolve()), 'manifest_name': MANIFEST_NAME,
            'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'split_seed': 42,
            'train_size': len(indices['train']), 'val_size': len(indices['dev']),
            'test_size': len(indices['test']), 'train_groups': 9000,
            'val_groups': 500, 'test_groups': 500}
    return indices, info

def require_fixed_provenance(info):
    if (info.get('protocol') != 'frozen_90_5_5' or info.get('manifest_name') != MANIFEST_NAME
            or [info.get(k) for k in ('train_size', 'val_size', 'test_size')] != [17847, 991, 991]
            or not re.fullmatch(r'[0-9a-f]{64}', str(info.get('manifest_sha256', '')))):
        raise ValueError('Missing fixed 90/5/5 provenance; retrain on train and select on dev')
    return info

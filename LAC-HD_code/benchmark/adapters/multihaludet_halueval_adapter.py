"""Export fixed train/dev/test files for an external MultiHaluDet integration.

This adapter does not launch an upstream random-split pipeline. Source-token
labels are used for export; scored-token alignment remains an integration check.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Code'))
from fixed_split import DEFAULT_SPLIT_MANIFEST, dialogue_group_id, fixed_split_indices

def export_splits(rows, manifest_path, output_dir):
    partitions, split_info = fixed_split_indices(rows, manifest_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, indices in partitions.items():
        converted = [{'id': str(rows[i]['id']), 'group_id': dialogue_group_id(rows[i]),
                      'question': str(rows[i].get('question', '')),
                      'answer': str(rows[i].get('answer', '')),
                      'is_hallucination': int(any(rows[i].get('token_labels', [])))} for i in indices]
        (output / f'{name}.json').write_text(json.dumps(converted, ensure_ascii=False), encoding='utf-8')
    (output / 'split_info.json').write_text(json.dumps(split_info, indent=2), encoding='utf-8')
    return split_info

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--split-manifest', default=DEFAULT_SPLIT_MANIFEST)
    args = parser.parse_args()
    rows = json.loads(Path(args.input).read_text(encoding='utf-8'))
    print(json.dumps(export_splits(rows, args.split_manifest, args.output_dir), indent=2))

if __name__ == '__main__':
    main()

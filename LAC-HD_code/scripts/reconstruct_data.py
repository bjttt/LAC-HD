"""Rebuild the supplied annotation derivative from user-obtained HaluEval text."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read_records(path):
    text = path.read_text(encoding='utf-8-sig')
    return json.loads(text) if text.lstrip().startswith('[') else [json.loads(line) for line in text.splitlines() if line.strip()]


def spans(tokens, labels):
    result = []
    pos = 0
    begin = None
    for token, label in zip(tokens, labels):
        if label == 1 and begin is None:
            begin = pos
        if label == 0 and begin is not None:
            result.append([begin, pos])
            begin = None
        pos += len(token)
    if begin is not None:
        result.append([begin, pos])
    return result


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records), encoding='utf-8')


def reconstruct(source_path, package, output):
    metadata = json.loads((package/'metadata.json').read_text(encoding='utf-8'))
    original = read_records(source_path)
    if len(original) * 2 != metadata['source_candidate_responses']:
        raise ValueError('Unexpected source row count; use the matching 10,000-row dialogue_data.json')
    annotations = read_records(package/'annotations/dialogue_annotations.jsonl')
    rows = []
    seen = set()
    for ann in annotations:
        idx = ann['source_row_index']
        item = original[idx]
        question = f"Knowledge: {item['knowledge']}\nDialogue history: {item['dialogue_history']}"
        answer = item[ann['source_response_key']]
        if sha(answer) != ann['answer_sha256'] or sha(question) != ann['question_sha256']:
            raise ValueError(f"Source content/order mismatch for {ann['id']}; do not reorder source rows")
        decoded = answer
        for patch in reversed(ann.get('decoded_answer_patches', [])):
            replacement = ''.join(chr(n) for n in patch['replacement_codepoints'])
            decoded = decoded[:patch['start']] + replacement + decoded[patch['end']:]
        if sha(decoded) != ann['decoded_answer_sha256']:
            raise ValueError(f"Decoded-answer checksum mismatch for {ann['id']}")
        tokens = []
        pos = 0
        lengths = ann['source_token_character_lengths']
        labels = ann['source_token_labels']
        if len(lengths) != len(labels) or any(n < 0 for n in lengths) or any(n not in (0,1) for n in labels):
            raise ValueError(f"Invalid token fields for {ann['id']}")
        for length in lengths:
            tokens.append(decoded[pos:pos+length])
            pos += length
        if pos != len(decoded) or spans(tokens, labels) != ann['hallucination_spans_legacy']:
            raise ValueError(f"Token reconstruction mismatch for {ann['id']}")
        if ann['id'] in seen:
            raise ValueError('Duplicate annotation ID')
        seen.add(ann['id'])
        rows.append({'id':ann['id'], 'question':question, 'answer':answer,
                     'label':ann['response_label'], 'source':ann['source_response_key'],
                     'tokens':tokens, 'token_labels':labels})
    if len(rows) != metadata['retained_responses']:
        raise ValueError('Annotation count mismatch')
    write_json(output/'halu_eval/token level/dialogue_token_hallucination.json', rows)
    grouped = {r['id']: r['id'].rsplit('_',1)[0] for r in rows}
    for protocol, filename in [('frozen_90_5_5','halueval_dialogue_group_90_5_5_seed42.json')]:
        manifest_path = package/'splits'/filename
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        assigned_ids = set()
        assigned_groups = set()
        for name, partition in manifest['partitions'].items():
            member_ids = set(partition['sample_ids'])
            group_ids = set(partition['group_ids'])
            if assigned_ids & member_ids or assigned_groups & group_ids:
                raise ValueError('Partition leakage')
            if member_ids != {sid for sid,g in grouped.items() if g in group_ids}:
                raise ValueError('Split membership mismatch')
            assigned_ids |= member_ids
            assigned_groups |= group_ids
            selected = [r for r in rows if r['id'] in member_ids]
            if len(selected) != partition['stats']['num_samples']:
                raise ValueError('Split sample count mismatch')
            write_json(output/'prepared'/protocol/f'{name}.json', selected)
            unified = []
            for r in selected:
                ranges = spans(r['tokens'],r['token_labels'])
                unified.append({'id':r['id'],'dataset':'halueval_token_dialogue',
                                'task':'dialogue_token_detection','prompt':r['question'],
                                'context':'','response':r['answer'],
                                'spans':[{'start':s,'end':e,'label':'hallucination'} for s,e in ranges],
                                'response_label':int(any(r['token_labels']) or r['label']==1),
                                'meta':{'source':r['source'],'sample_label':r['label'],'split':name,
                                        'annotation_coordinates':'legacy_concatenated_token_decodes'}})
            write_jsonl(output/'prepared'/protocol/f'{name}.unified.jsonl',unified)
        if assigned_ids != seen:
            raise ValueError('Split does not cover every annotation')
    report = {'reconstructed_records':len(rows),'all_retained_question_answer_hashes_verified':True,
              'token_lengths_and_spans_verified':True,'group_disjoint_partitions_verified':True,
              'original_file_sha256_matches':hashlib.sha256(source_path.read_bytes()).hexdigest()==metadata['source_file_sha256'],
              'source_token_ids_omitted':True,'tokenizer_download_required':False,
              'preserves_legacy_decode_mismatches':metadata['quality']['source_token_decode_mismatch_records']}
    write_json(output/'reconstruction_report.json', report)
    print(json.dumps(report, indent=2))


def main():
    base = Path(__file__).resolve().parents[1]
    default_package = base if (base/'annotations').exists() else base.parent/'LAC-HD_data'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='User-obtained HaluEval dialogue_data.json')
    parser.add_argument('--data_package', type=Path, default=default_package)
    parser.add_argument('--output_dir', required=True, type=Path)
    args = parser.parse_args()
    reconstruct(args.source, args.data_package, args.output_dir)


if __name__ == '__main__':
    main()

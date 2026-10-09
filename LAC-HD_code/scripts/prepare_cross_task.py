"""Rebuild the exact supplied cross-task subsets from independently obtained HaluEval."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmark.adapters.halueval_response_adapter import convert_qa, convert_summarization, convert_general

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source_dir',required=True,type=Path)
    p.add_argument('--data_package',type=Path,default=ROOT.parent/'LAC-HD_data')
    p.add_argument('--output_dir',type=Path,default=ROOT/'benchmark/results/raw/halueval')
    args=p.parse_args()
    manifest=json.loads((args.data_package/'cross_task_sources.json').read_text(encoding='utf-8'))
    converters={'qa':convert_qa,'summarization':convert_summarization,'general':convert_general}
    report={}
    for entry in manifest:
        selected=[json.loads(line) for line in (args.data_package/entry['selection_file']).read_text(encoding='utf-8').splitlines() if line.strip()]
        needed={r['source_row_index'] for r in selected}
        source=args.source_dir/entry['source_file']
        text=source.read_text(encoding='utf-8-sig')
        source_rows={}
        if text.lstrip().startswith('['):
            source_rows={i:row for i,row in enumerate(json.loads(text)) if i in needed}
        else:
            for i,line in enumerate(text.splitlines()):
                if i in needed:
                    try: source_rows[i]=json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f'Selected source row {i} in {source.name} is incomplete') from exc
        records=[]
        for selection in selected:
            idx=selection['source_row_index']
            if idx not in source_rows: raise ValueError(f'Missing selected row {idx} in {source.name}')
            row=source_rows[idx]
            canonical=json.dumps(row,sort_keys=True,ensure_ascii=False,separators=(',',':'))
            actual=hashlib.sha256(canonical.encode('utf-8')).hexdigest()
            if actual!=selection['source_record_sha256']:
                raise ValueError(f'Content/order mismatch: {source.name}, row {idx}')
            records.append(row)
        converted=converters[entry['name']](records)
        if len(converted)!=entry['response_count']: raise ValueError('Response count differs from manifest')
        args.output_dir.mkdir(parents=True,exist_ok=True)
        output=args.output_dir/f"{entry['name']}_response.unified.jsonl"
        output.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in converted),encoding='utf-8')
        report[entry['name']]={'source_records':len(records),'responses':len(converted),
                               'positive_responses':sum(r['response_label']==1 for r in converted),
                               'all_selected_source_hashes_verified':True}
    (args.output_dir/'cross_task_reconstruction_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()

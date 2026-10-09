"""Evaluate with thresholds selected once on source Dialogue development data."""
from __future__ import annotations
import argparse
import json
import math
import hashlib
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'Code'))
from fixed_split import DEFAULT_SPLIT_MANIFEST, require_fixed_provenance

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_path',required=True)
    p.add_argument('--ckpt',required=True)
    p.add_argument('--dev_metrics',required=True,type=Path)
    p.add_argument('--eval_jsonl',required=True)
    p.add_argument('--output_json',required=True)
    p.add_argument('--response_labels_jsonl',default='',help='Shared paper test response labels; checked against valid scored tokens')
    p.add_argument('--mode',choices=['dialogue','cross_task'],default='dialogue')
    p.add_argument('--dtype',choices=['auto','bf16','fp16','fp32'],default='auto')
    p.add_argument('--device',default='cuda')
    p.add_argument('--limit',type=int,default=0,help='Debug only; 0 evaluates all rows')
    p.add_argument('--split_manifest',default=DEFAULT_SPLIT_MANIFEST)
    p.add_argument('--dry_run',action='store_true')
    args=p.parse_args()
    metrics_path=args.dev_metrics if args.dev_metrics.is_absolute() else ROOT/args.dev_metrics
    metrics=json.loads(metrics_path.read_text(encoding='utf-8'))
    info = require_fixed_provenance(metrics.get('split_info', {}))
    manifest_bytes = Path(args.split_manifest).read_bytes()
    manifest = json.loads(manifest_bytes.decode('utf-8'))
    if info['manifest_sha256'] != hashlib.sha256(manifest_bytes).hexdigest():
        p.error('Dev metrics came from a different split manifest')
    dev_path = Path(metrics['eval_jsonl'])
    if not dev_path.is_absolute(): dev_path = ROOT/dev_path
    dev_ids = [json.loads(line)['id'] for line in dev_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(dev_ids) != 991 or set(dev_ids) != set(manifest['partitions']['dev']['sample_ids']):
        p.error('Threshold source must evaluate exactly the fixed Dialogue dev partition')
    if Path(metrics['ckpt']).resolve() != Path(args.ckpt).resolve() or metrics.get('model_path') != args.model_path:
        p.error('Dev thresholds must come from the matching backbone and checkpoint')
    token_t=float(metrics['token']['best_threshold'])
    response_t=float(metrics['response']['best_threshold'])
    if not math.isfinite(token_t) or not math.isfinite(response_t):
        p.error('Development thresholds must be finite')
    cmd=[sys.executable,'-m','benchmark.baselines.confidence_head_runner',
         '--model_path',args.model_path,'--ckpt',args.ckpt,
         '--eval_jsonl',args.eval_jsonl,'--output_json',args.output_json,
         '--max_length','512','--limit',str(args.limit),'--dtype',args.dtype,'--device',args.device,
         '--response_agg','max','--threshold',str(token_t),'--response_threshold',str(response_t),
         '--response_label_source','evaluated_tokens' if args.mode=='dialogue' else 'dataset',
         '--truncation_side','right' if args.mode=='dialogue' else 'left']
    if args.response_labels_jsonl:
        if args.mode != 'dialogue': p.error('Shared token-derived test labels apply to dialogue mode')
        cmd += ['--response_labels_jsonl',args.response_labels_jsonl]
    print(json.dumps({'threshold_source':str(metrics_path),'command':cmd},indent=2),flush=True)
    if not args.dry_run: subprocess.run(cmd,cwd=ROOT,check=True)

if __name__=='__main__':
    main()

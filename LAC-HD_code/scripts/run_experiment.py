"""Run the packaged trainer with an explicit, portable experiment configuration."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--model_path',default='')
    parser.add_argument('--data_package',type=Path,default=ROOT.parent/'LAC-HD_data')
    parser.add_argument('--head_type',choices=['base','no_conv','deep_mlp','last_layer_linear','mean_layer_mlp'])
    parser.add_argument('--output_dir',default='')
    parser.add_argument('--dry_run',action='store_true')
    args=parser.parse_args()
    config_path=args.config if args.config.is_absolute() else ROOT/args.config
    cfg=json.loads(config_path.read_text(encoding='utf-8'))
    if cfg.get('protocol') != 'frozen_90_5_5':
        parser.error('Only the fixed paper 90/5/5 protocol is supported')
    options=dict(cfg['train_args'])
    if args.model_path: options['model_path']=args.model_path
    if args.head_type: options['head_type']=args.head_type
    if args.output_dir: options['output_dir']=args.output_dir
    options['split_manifest']=options['split_manifest'].format(data_package=args.data_package.resolve().as_posix())
    if args.head_type and not args.output_dir:
        options['output_dir'] += '_'+args.head_type
    cmd=[sys.executable,str(ROOT/'Code/train_confidence_head.py')]
    for key,value in options.items():
        if isinstance(value,bool):
            if value: cmd.append('--'+key)
        else: cmd += ['--'+key,str(value)]
    print(json.dumps({'configuration':cfg['name'],'command':cmd},indent=2),flush=True)
    if not args.dry_run:
        data_path=Path(options['train_json'])
        if not data_path.is_absolute(): data_path=ROOT/data_path
        if not data_path.is_file() or not Path(options['split_manifest']).is_file():
            parser.error('Run scripts/reconstruct_data.py first and place LAC-HD_data beside LAC-HD_code')
        subprocess.run(cmd,cwd=ROOT,check=True)

if __name__=='__main__':
    main()

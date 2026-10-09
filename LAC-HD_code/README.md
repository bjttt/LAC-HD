## Contents

- Code/: cross-layer attention, residual Conv1D, token MLP, structural controls, dataset alignment, training and prediction.
- configs/: Qwen3-4B, Qwen3-8B and Llama-2-7b configurations; the paper protocol uses frozen 90/5/5.
- LLAMA7B_README.md: Llama-specific head, tokenizer handling, training and evaluation commands.
- scripts/: data reconstruction, fixed-membership training, cross-task reconstruction, frozen-threshold evaluation.
- benchmark/: token/span/response metrics, max pooling, data adapters and baseline runners.
- benchmark-8B/: original shared-backbone ablation implementation.
- requirements.txt / requirements_optional.txt: core and optional dependencies.
- REPRODUCTION_STATUS.md: evidence and missing experiment metadata.
- PACKAGING_CHANGES.md: changes made only in this supplement.

## Environment and backbones

Use Python 3.10 or newer and a CUDA GPU suitable for the chosen frozen backbone. Install a PyTorch build matching your CUDA environment, then:

~~~bash
python -m pip install -r requirements.txt
~~~

Dependency ranges were assembled from source imports; the original experiment environment lockfile was not supplied. GPU training was not rerun while preparing this supplement.

The two models evidenced in this project are [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) and [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B). Obtain the original model and tokenizer from these repositories or supply a local snapshot with --model_path. Do not substitute a differently named later model variant. Model/tokenizer revision hashes were not recorded in the original experiments. Llama-2-7b support has now been added with the confirmed base model [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf); see LLAMA7B_README.md.

The Qwen backbones have 36 Transformer layers with hidden sizes 2560 and 4096. Llama-2-7b uses the dedicated 32-layer, 4096-feature head described in LLAMA7B_README.md. The embedding output is excluded. Backbone parameters are frozen and only the detection head is trained.

## Reconstruct the token dataset

Obtain the source files from [HaluEval](https://github.com/RUCAIBox/HaluEval). From this software directory:

~~~bash
python scripts/reconstruct_data.py --source /path/to/HaluEval/data/dialogue_data.json --output_dir data
~~~

The script verifies every retained question/answer hash and reconstructs all 19,829 records without downloading a tokenizer or making annotation API calls. It writes the dataset path plus partition JSON and unified JSONL files under data/prepared/frozen_90_5_5/. Legacy source token IDs are unnecessary for the loader and are omitted; source token strings and labels are reconstructed exactly.

## Train the detection head

The paper protocol trains and evaluates using the fixed 90/5/5 dialogue-group manifests:

~~~bash
python scripts/run_experiment.py --config configs/qwen3_4b_frozen_90_5_5.json
python scripts/run_experiment.py --config configs/qwen3_8b_frozen_90_5_5.json
python scripts/run_experiment.py --config configs/llama2_7b_frozen_90_5_5.json
~~~

Optional arguments: --model_path /path/to/model, --head_type no_conv (or deep_mlp / last_layer_linear / mean_layer_mlp), --output_dir /path/to/output, --data_package /path/to/LAC-HD_data, and --dry_run. Fixed split membership is independent of training seed. Core training uses seed 42, three epochs, maximum length 512, batch size 1, eight-step accumulation, AdamW learning rate 1e-4, weight decay 0.01, warmup 0.05 and dropout 0.1. The positive-class weight is computed from the actual training partition; do not copy a legacy weight into a new partition.

The original 8B ablations train multiple heads with shared backbone forwards. To preserve that particular execution order:

~~~bash
python benchmark-8B/train_ablation_heads.py --model_path Qwen/Qwen3-8B --split_manifest ../LAC-HD_data/splits/halueval_dialogue_group_90_5_5_seed42.json --variants base no_conv deep_mlp mean_layer_mlp last_layer_linear --epochs 3 --max_length 512 --batch_size 1 --grad_accum 8 --dtype bf16 --seed 42
~~~

A separately trained head and a head trained alongside other variants can consume different random-number sequences. Exact numerical equality is not promised without the original checkpoints and environment.

## Evaluate development and frozen test

Example for the frozen 4B experiment, run from this directory:

~~~bash
python -m benchmark.baselines.confidence_head_runner --model_path Qwen/Qwen3-4B --ckpt Checkpoints/qwen3_4b_frozen_90_5_5/best.pt --eval_jsonl data/prepared/frozen_90_5_5/dev.unified.jsonl --output_json results/qwen3_4b_dev.json --max_length 512 --response_agg max --response_label_source evaluated_tokens --truncation_side right
python scripts/evaluate_frozen.py --model_path Qwen/Qwen3-4B --ckpt Checkpoints/qwen3_4b_frozen_90_5_5/best.pt --dev_metrics results/qwen3_4b_dev.json --eval_jsonl data/prepared/frozen_90_5_5/test.unified.jsonl --output_json results/qwen3_4b_test.json --mode dialogue --response_labels_jsonl ../LAC-HD_data/evaluation/test_response_labels.jsonl
~~~

Select checkpoints and thresholds on dev. Retrain each head on the fixed train partition. Previous two-way-split checkpoints cannot support independent-test claims.

Token probabilities are sigmoid(logit); response scores are the maximum over evaluated response tokens. The matched Dialogue comparison defines response gold as the presence of at least one positive evaluated token after truncation. Use --response_labels_jsonl to check exact test membership and agreement of valid scored-token aggregation with shared gold; a mismatch stops evaluation for alignment inspection. All backbones and methods use the same 991 test response labels: 491/500 original positive/negative responses become 463/528 after 28 positive-to-negative changes and no negative-to-positive changes. Shared gold is provided by the data supplement in evaluation/test_response_labels.jsonl. The whole-dataset count of 536 all-zero original-positive records must not be used as a test-set count. Prompt, padding and zero-length offsets are ignored with label -100.

The output token/response best_f1 values are oracle threshold-sweep diagnostics on that evaluation set. Use token_at_frozen_threshold and response_fixed for held-out F1 claims. AUROC and AUPRC are threshold-free. Target-span metrics apply to HaDeS; token metrics on datasets with only response gold are not valid localization results.

## Frozen cross-task evaluation

The supplied project used 20,000 QA responses, 11,404 Summarization responses and 4,507 General responses. The data supplement fixes those source rows and their hashes. The local Summarization source ended with one incomplete JSON row; only its 5,702 complete rows are selected. Fetching a complete upstream dataset does not expand this evaluation subset.

~~~bash
python scripts/prepare_cross_task.py --source_dir /path/to/HaluEval/data
python scripts/evaluate_frozen.py --model_path Qwen/Qwen3-4B --ckpt Checkpoints/qwen3_4b_frozen_90_5_5/best.pt --dev_metrics results/qwen3_4b_dev.json --eval_jsonl benchmark/results/raw/halueval/qa_response.unified.jsonl --output_json results/qa.json --mode cross_task
~~~

Create results/qwen3_4b_dev.json with the development evaluation command above using the matching fixed-split checkpoint and partition. Repeat the second command for summarization_response.unified.jsonl and general_response.unified.jsonl. Target evaluation uses left truncation at 512 tokens, upstream response labels, max pooling and source-development thresholds. Do not tune target-task thresholds. For newly trained frozen-split models, use that model's own Dialogue development metrics instead. The cross-task shell launcher also reads matching development thresholds and defaults to the confidence head. Optional UQ/ICR transfer requires new fixed-split checkpoints and their own dev thresholds.

## Other benchmarks and optional baselines

Optional dependencies:

~~~bash
python -m pip install -r requirements_optional.txt
~~~

HaDeS preparation/evaluation code is in Code/prepare_hades.py, benchmark/adapters/hades_adapter.py and benchmark/scripts/run_hades.sh. Obtain HaDeS independently. Its designated replacement target is the primary sample-level task; full-response pooling is a separate diagnostic.

UQ Head requires separately obtaining [llm-uncertainty-head](https://github.com/IINemo/llm-uncertainty-head) and [lm-polygraph](https://github.com/IINemo/lm-polygraph). Original external repositories and their revision hashes were not supplied. MultiHaluDet's fitted fold models/scalers/stacker were not supplied, so its existing frozen-transfer results cannot be reconstructed without retraining. Local token-feature probes, confidence-head evaluators, shared 8B training and the UQ dataset adapter now use the same strict fixed manifest. UQ training receives train and validation only; reserved test responses are excluded from that cache. The optional external UQ trainer is not bundled or integration-tested. MultiHaluDet automatic two-way-split launchers have been removed; its adapter exports separate train/dev/test JSON files for an external fixed-split integration. Collectors report development diagnostics and reject results without fixed-split provenance. Consult REPRODUCTION_STATUS.md before using an optional baseline for a paper claim.

Original absolute workstation paths were replaced with portable defaults. For legacy Bash runners, set ROOT to the absolute path of this software folder and override MODEL_PATH / checkpoint paths as needed. The Python launchers are the primary portable entry points.

## Data and use terms

See the data supplement README and TERMS.md. HaluEval's repository identifies its license as MIT; retain the upstream license/citation when obtaining or redistributing its resources. This package does not assign a new license to project-authored code or annotations; the original project did not provide one. The paper specifies GPT-5.6-Sol pre-annotation and full independent review by two reviewers; follow ANNOTATION_PROTOCOL.md.

## Code inventory and split migration

See CODE_GUIDE_zh.md for every Python and Shell file's purpose, and SPLIT_MIGRATION_zh.md for the migration scope and validation limits. Training entry points no longer accept two-way split arguments or regenerate membership. The fixed sample counts are train 17,847, dev 991, test 991; dialogue-group counts are 9,000/500/500.

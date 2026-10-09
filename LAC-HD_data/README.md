# LAC-HD data supplement

This package contains the token annotations and fixed memberships for the HaluEval-Dialogue derivative. The authoritative annotation and evaluation policy is the author-supplied paper Appendix A, reproduced in ANNOTATION_PROTOCOL.md. It contains no original question, context, response text, source token strings or source token IDs. Obtain the source text from [HaluEval](https://github.com/RUCAIBox/HaluEval) and reconstruct it with the included script.

## Contents and counts

- annotations/dialogue_annotations.jsonl: 19,829 retained responses (10,000 original factual responses and 9,829 original hallucinated responses).
- splits/halueval_dialogue_group_90_5_5_seed42.json: frozen train/dev/test manifest.
- splits/halueval_dialogue_group_95_5_seed42_legacy.json: memberships used by existing development results.
- splits/frozen_90_5_5/ and splits/legacy_95_5/: separate sample-ID and dialogue-group-ID lists.
- filtering/excluded_candidates.jsonl: 171 candidate IDs excluded before splitting for unresolved labels or unreliable data quality, following Appendix A.2.
- evaluation/: shared 991 test response labels; 28 original positives become negative, giving 463 positives and 528 negatives.
- quality/: source-token alignment details and full-dataset token/source-label comparison lists.
- cross_task/: selected source-row indices and hashes for QA, Summarization and General evaluation.
- metadata.json / cross_task_sources.json: provenance, counts, source hashes and documented gaps.
- scripts/reconstruct_data.py: standard-library reconstruction; no annotation API or tokenizer download.

| Protocol | Partition | Groups | Responses |
|---|---|---:|---:|
| Frozen 90/5/5 | train | 9,000 | 17,847 |
| Frozen 90/5/5 | dev | 500 | 991 |
| Frozen 90/5/5 | test | 500 | 991 |
| Legacy 95/5 | train | 9,500 | 18,838 |
| Legacy 95/5 | dev | 500 | 991 |

The paper protocol uses the fixed 90/5/5 dialogue-group split after filtering. All backbones and methods share the same 991 test responses and final response gold labels. The legacy 95/5 membership is retained for historical implementation reference only; it must not replace the paper split. Both memberships share dev groups, and legacy training includes the frozen test groups.

## Annotation fields

Each JSONL record has:

| Field | Meaning |
|---|---|
| id | dialogue_NNNNNN_right or dialogue_NNNNNN_hallucinated |
| group_id | dialogue_NNNNNN shared by both response variants |
| source_row_index | Zero-based source-file record index; preserve source order |
| source_response_key | right_response or hallucinated_response |
| response_label | Original response-level label (0 factual, 1 hallucinated) |
| source_token_labels | 0 supported/non-hallucinatory token, 1 hallucination token |
| source_token_character_lengths | Length of each independently decoded source token in Python Unicode codepoints |
| hallucination_spans_legacy | Contiguous positive ranges as [start,end) in concatenated decoded source tokens |
| answer_sha256 / question_sha256 | SHA-256 of exact UTF-8 response and assembled question |
| decoded_answer_sha256 | SHA-256 of the concatenated independently decoded source tokens |
| decoded_answer_patches | Optional patches from original answer to legacy decoded answer; replacement codepoints are U+FFFD only |

The source tokenizer is inferred as Qwen/Qwen3-4B from the supplied generation script. Its exact revision was not recorded. Answers were encoded without special tokens, then each token was independently decoded with clean_up_tokenization_spaces=False. The reconstruction script uses lengths, original text and the optional numeric patches to reproduce source token strings exactly without depending on a tokenizer revision.

Coordinates are zero-based, half-open Python Unicode codepoint indices, not UTF-8 bytes or UTF-16 indices. For 19,807 records the decoded token concatenation equals the original answer. For 22 records independent byte-fragment decoding introduced U+FFFD replacement characters, so the legacy coordinates refer to decoded token concatenation and may differ from original-answer coordinates. Their IDs are listed explicitly. No labels or spans were silently corrected in this release. Software reproduces the existing legacy alignment; a future corrected alignment requires a separately versioned dataset and rerun results.

The training loader shifts response-local spans by the question-prefix length and marks a backbone token positive if its offset overlaps a positive span. Prompt/padding/zero-length offsets receive -100. Training and Dialogue evaluation use right truncation at 512 total input tokens; cross-task evaluation uses left truncation.

## Reconstruction

Place LAC-HD_code and LAC-HD_data beside each other. From this data folder:

~~~bash
python scripts/reconstruct_data.py --source /path/to/HaluEval/data/dialogue_data.json --output_dir ../LAC-HD_code/data
~~~

All retained response and question hashes must match. The script restores id, question, answer, label, source, tokens and token_labels, verifies spans and group-disjoint partition coverage, and writes partition JSON/unified JSONL files. Original source token IDs are not restored because the training loader retokenizes full input using the chosen backbone tokenizer.

The frozen manifest's old source_sha256 did not match the current annotation file bytes. Membership and all partition statistics were independently checked and retained. The package records the current input hash and preserves the old value as original_manifest_source_sha256. Reconstructed JSON formatting produces different file bytes; use record-level checks rather than expecting the reconstructed file hash to equal the historical input hash.

## Annotation guidelines, review and filtering

See ANNOTATION_PROTOCOL.md for the clean Appendix A text. Supplied knowledge is the primary factual evidence; dialogue history resolves context and references. An auxiliary reference response cannot override knowledge. External world knowledge is excluded. Label 1 marks incorrect, fabricated, unsupported or contradictory factual content; all other tokens receive 0.

Annotators mark the smallest contiguous content-bearing spans, including erroneous entities, dates, numbers, attributes and relations. For an incorrect relation, label its content-bearing tokens. Function words and punctuation normally receive 0. Hedging does not exempt an unsupported factual claim. Preserve the supplied token sequence with exactly one binary label per token.

GPT-5.6-Sol pre-annotates all 10,000 original hallucinated candidates. Two reviewers independently inspect and correct token labels and span boundaries using the knowledge and dialogue history. Disagreements are discussed under the shared guidelines. Before splitting, 171 candidates are excluded for unresolved labels or data quality that prevents reliable annotation; 9,829 candidates remain. Both reviewers inspect all 10,000 reference responses and confirm their source-derived all-zero labels. Reference responses are separate detector examples and are not appended to candidate scoring inputs.

The exclusion file records those paper-defined categories. The combined category is recorded for each known excluded ID; the supplied per-ID records do not distinguish the two subcategories.

## Test response labels

The same 991 test responses are used for all backbones and methods. Original source labels contain 491 positives and 500 negatives. After human revision and alignment, 28 original positives have no positive labels among valid scored tokens and become negative; 463 positives remain and no original negatives become positive. Final test labels are therefore 463 positive and 528 negative.

The test-label file and 28 changed IDs are provided in evaluation/. The current annotations yield exactly this transition count when aggregating source-token labels. Packaging did not rerun GPU backbone alignment: each evaluation must use the valid scored-token mask and check against the shared response-label file. The 536 all-zero original-positive records in quality/ are a whole-dataset count, not the test-set count.

## Cross-task memberships

QA selects 10,000 source rows (20,000 paired responses), Summarization selects 5,702 complete source rows (11,404 paired responses), and General selects 4,507 source rows/responses. The supplied Summarization file ends with one incomplete JSON row at source index 5702. That row is not selected. These lists reproduce the current project subsets; they are not a claim that the full upstream benchmark has these sizes.

Use LAC-HD_code/scripts/prepare_cross_task.py with independently obtained upstream files. Every selected source-record hash is checked before conversion. Labels remain upstream response-level gold, with the General field named hallucination in the supplied data. Membership/hash records include no original text.

## Source and terms

The upstream repository identifies HaluEval's license as MIT and identifies OpenDialKG as the Dialogue seed dataset. See [upstream documentation](https://github.com/RUCAIBox/HaluEval#data-release) and [license](https://github.com/RUCAIBox/HaluEval/blob/main/LICENSE). Cite HaluEval when using these annotations. See TERMS.md for the unrecorded project annotation license and the distinction between upstream text and derived labels.

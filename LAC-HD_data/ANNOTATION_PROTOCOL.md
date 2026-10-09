# Data Annotation and Evaluation Protocol

This is the authoritative protocol supplied by the author from Appendix A of the paper. It supersedes annotation provenance descriptions in older local project notes.

## A.1 Annotation Guidelines

**Evidence and labels.** The supplied knowledge is the primary factual evidence; dialogue history resolves context and references. The reference response may serve as an auxiliary aid but cannot override the knowledge. External world knowledge is excluded. Label 1 marks incorrect, fabricated, unsupported, or contradictory factual content; other tokens receive 0.

**Span boundaries.** Annotators mark the smallest contiguous spans carrying unsupported content, including erroneous entities, dates, numbers, attributes, and relations. For an incorrect relation, its content-bearing tokens are marked. Function words and punctuation normally receive 0; hedging does not exempt an unsupported factual claim. The supplied token sequence remains unchanged, with one binary label per token.

## A.2 Human Review and Filtering

GPT-5.6-Sol produces token pre-annotations for all 10,000 original hallucinated candidates. Two reviewers independently inspect the labels and span boundaries against the knowledge and dialogue history, correcting missed hallucinations, false positive labels, and boundary errors. Disagreements are discussed using the shared guidelines.

Before data splitting, 171 candidates are excluded because their labels remain unresolved or their data quality prevents reliable annotation, leaving 9,829 candidates. Both reviewers also inspect all 10,000 reference responses and confirm their source-derived all-zero token labels.

References are separate detector examples, not additional inputs when scoring their paired candidates. An auxiliary reference used during annotation must not be appended to a candidate's detector input.

## A.3 Data Splits and Response Labels

After filtering, responses are partitioned by dialogue group using the fixed 90/5/5 split. All backbones and methods share the same 991 test responses and response-level labels aggregated from valid scored tokens.

The test source labels comprise 491 positives and 500 negatives. After human revision and alignment, 28 originally positive responses have no positive labels among their valid scored tokens and receive negative response labels. The remaining 463 positives retain their labels, and no negatives become positive, yielding 463 positive and 528 negative test responses.

| Test-label stage | Positive | Negative | Total |
|---|---:|---:|---:|
| Original source response label | 491 | 500 | 991 |
| Final evaluation response label | 463 | 528 | 991 |

## Artifact checks

The packaged fixed test membership contains exactly 991 records. Its source-token labels identify the same 28 positive-to-negative IDs and yield 463 positives and 528 negatives. Their per-ID labels are stored in evaluation/test_response_labels.jsonl in the data supplement. This is a check of the supplied annotations and the author's paper protocol; GPU backbone alignment/inference was not rerun during packaging. Response gold must be checked against the valid scored-token mask when running each backbone. An alignment mismatch must be investigated, not silently resolved by changing the shared gold labels.

The 536 all-zero original-positive records counted across the complete 19,829-record dataset are not the test-set transition count. Per-token decode differences in 22 records are implementation alignment details, not alternative annotation guidelines.

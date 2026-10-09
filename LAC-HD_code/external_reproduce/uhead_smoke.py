"""Minimal smoke test for IINemo/llm-uncertainty-head.

This script intentionally keeps the call path defensive because the official
package API may change. It verifies that the package, base LLM, and uncertainty
head can be loaded, then tries a small set of common inference entrypoints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from uhead_import_utils import prepare_uhead_imports


def to_jsonable(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return float(x.detach().cpu().item())
        return x.detach().cpu().float().tolist()
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if hasattr(x, "__dict__"):
        return {"type": type(x).__name__, "dict": to_jsonable(vars(x))}
    try:
        json.dumps(x)
        return x
    except TypeError:
        return repr(x)


@torch.no_grad()
def try_predict(llm: Any, uhead: Any, tokenizer: Any, prompt: str, device: str) -> dict[str, Any]:
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = llm(
        **enc,
        output_attentions=bool(getattr(uhead, "output_attentions", False)),
        output_hidden_states=True,
        use_cache=False,
    )
    # Treat the whole prompt as the scored continuation except the first token.
    out["context_lengths"] = torch.tensor([1], device=device)
    enc["context_lengths"] = out["context_lengths"]
    logits = uhead(enc, out).detach().float().cpu().squeeze(0).squeeze(-1)
    scores = torch.sigmoid(logits).tolist()
    return {
        "entrypoint": "teacher_forcing_uhead_forward",
        "num_scores": len(scores),
        "scores_first_20": scores[:20],
        "mean_score": float(torch.tensor(scores).mean().item()) if scores else 0.0,
        "max_score": float(torch.tensor(scores).max().item()) if scores else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--uhead_name", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--output_json", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    args = p.parse_args()

    prepare_uhead_imports(THIS_DIR.parents[0])
    try:
        from luh import AutoUncertaintyHead  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "Cannot import `luh.AutoUncertaintyHead`. Install official repo first, e.g.\n"
            "  pip install git+https://github.com/IINemo/lm-polygraph.git@dev\n"
            "  pip install git+https://github.com/IINemo/llm-uncertainty-head.git\n"
            f"Original error: {e!r}"
        )

    dtype = None
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp32":
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model_kwargs = {"trust_remote_code": True}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    llm = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs).to(args.device)
    llm.eval()

    uhead = AutoUncertaintyHead.from_pretrained(args.uhead_name, base_model=llm)
    if hasattr(uhead, "to"):
        uhead = uhead.to(args.device)
    if hasattr(uhead, "eval"):
        uhead.eval()

    result = {
        "model_name": args.model_name,
        "uhead_name": args.uhead_name,
        "prompt": args.prompt,
        "device": args.device,
        "result": try_predict(llm, uhead, tokenizer, args.prompt, args.device),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

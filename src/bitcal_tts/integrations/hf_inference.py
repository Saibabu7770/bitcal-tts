"""
Hugging Face causal LM integration: single forward pass for logits and hidden states.

Requires: pip install bitcal-tts[research]
Optional 4/8-bit: bitsandbytes + compatible GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import torch

LoadCausalLMFn = Callable[..., Tuple[Any, Any]]


def transformers_available() -> bool:
    try:
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass
class ForwardResult:
    logits_last: torch.Tensor
    hidden_last: Optional[torch.Tensor]
    prompt: str


def load_causal_lm(
    model_name_or_path: str,
    *,
    device_map: str | dict[str, Any] | None = "auto",
    torch_dtype: torch.dtype | str | None = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    trust_remote_code: bool = False,
) -> Tuple[Any, Any]:
    """Load AutoModelForCausalLM + tokenizer. Uses float32 on CPU if no GPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if load_in_4bit and load_in_8bit:
        raise ValueError("Specify at most one of load_in_4bit and load_in_8bit")

    if torch_dtype is None:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    quant_config = None
    if load_in_4bit or load_in_8bit:
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
    }
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    return model, tokenizer


def forward_last_logits(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    output_hidden_states: bool = True,
) -> ForwardResult:
    """One forward on prompt; returns logits at last position and last-layer hidden state."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=output_hidden_states)
    logits = out.logits[0, -1]
    hidden = None
    if output_hidden_states and getattr(out, "hidden_states", None) is not None:
        hidden = out.hidden_states[-1][0, -1]
    return ForwardResult(logits_last=logits, hidden_last=hidden, prompt=prompt)


def hf_smoke_forward(
    model_name: str,
    prompt: str,
    *,
    load_fn: Optional[LoadCausalLMFn] = None,
    device_map: str | None | dict[str, Any] = None,
) -> ForwardResult:
    """
    Load model and run one forward; intended for CLI smoke tests.

    Parameters
    ----------
    load_fn
        Override for testing (inject mock loader). Defaults to load_causal_lm.
    device_map
        If None, uses ``\"auto\"`` on CUDA else CPU placement.
    """
    if not transformers_available():
        raise RuntimeError("Install transformers: pip install bitcal-tts[research]")

    loader = load_fn or load_causal_lm
    if device_map is None:
        device_map = "auto" if torch.cuda.is_available() else None

    model, tokenizer = loader(
        model_name,
        load_in_4bit=False,
        load_in_8bit=False,
        device_map=device_map,
    )
    if not torch.cuda.is_available():
        model = model.to(torch.device("cpu"))

    return forward_last_logits(model, tokenizer, prompt)


def hf_smoke_report(model_name: str, prompt: str, *, load_fn: Optional[LoadCausalLMFn] = None) -> ForwardResult:
    """Run hf_smoke_forward and print entropy summary (CLI)."""
    from bitcal_tts.signals.entropy import token_entropy

    res = hf_smoke_forward(model_name, prompt, load_fn=load_fn)
    ent = token_entropy(res.logits_last)
    print(f"model={model_name!r} prompt={prompt!r}")
    print(f"  last-token entropy (nats): {ent:.4f}")
    if res.hidden_last is not None:
        print(f"  hidden dim: {res.hidden_last.shape}")
    return res

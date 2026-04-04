"""
Real token-by-token generation with KV cache for BitCal-TTS experiments.

Implements HFStepGenerator: a stateful generator compatible with
run_fixed_budget_loop().  Requires pip install bitcal-tts[research].
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch

from bitcal_tts.runner.budget import TokenBudget


def _infer_input_device(model: Any) -> torch.device:
    """Best-effort: get device for first model parameter (handles device_map='auto')."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HFStepGenerator:
    """
    Stateful, step-by-step causal LM generator using the HF KV cache.

    Each call to `step()` generates `step_size` tokens (or fewer at end-of-budget),
    and returns (text_chunk, last_token_logits, last_hidden_state, n_tokens_generated).

    Compatible with run_fixed_budget_loop() as its step_fn argument.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        *,
        step_size: int = 16,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.95,
        seed: int = 42,
        output_hidden_states: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> None:
        if step_size < 1:
            raise ValueError(f"step_size must be >= 1, got {step_size}")

        self.model = model
        self.tokenizer = tokenizer
        self.step_size = step_size
        self.do_sample = do_sample
        self.temperature = max(temperature, 1e-6)
        self.top_p = top_p
        self.output_hidden_states = output_hidden_states

        self._device = _infer_input_device(model)
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(seed)

        enc = tokenizer(prompt, return_tensors="pt")
        self._current_ids: torch.Tensor = enc["input_ids"].to(self._device)
        self._attn_mask: torch.Tensor = enc.get(
            "attention_mask",
            torch.ones_like(self._current_ids),
        ).to(self._device)

        self._past_kv: Optional[Any] = None
        self._generated_ids: List[int] = []

        self._eos_id = eos_token_id
        if self._eos_id is None:
            self._eos_id = getattr(tokenizer, "eos_token_id", None)

        self.finished: bool = False

    def step(
        self,
        _step_idx: int,
        budget: TokenBudget,
    ) -> Tuple[str, torch.Tensor, Optional[torch.Tensor], int]:
        """
        Generate up to step_size tokens.

        Returns
        -------
        text_chunk   : newly decoded text
        last_logits  : logits for the last generated token  (vocab-size,)
        last_hidden  : last-layer hidden state at last token, or None
        n_tokens     : number of tokens generated this step
        """
        if self.finished:
            dummy_logits = torch.zeros(
                self.tokenizer.vocab_size or 32000, device=self._device
            )
            return "", dummy_logits, None, 0

        n_generated = 0
        last_logits: Optional[torch.Tensor] = None
        last_hidden: Optional[torch.Tensor] = None
        new_ids: List[int] = []

        for _ in range(self.step_size):
            if budget.exhausted():
                break

            with torch.no_grad():
                out = self.model(
                    input_ids=self._current_ids,
                    attention_mask=self._attn_mask,
                    past_key_values=self._past_kv,
                    use_cache=True,
                    output_hidden_states=self.output_hidden_states,
                )

            last_logits = out.logits[0, -1].detach()
            self._past_kv = out.past_key_values

            if self.output_hidden_states and getattr(out, "hidden_states", None) is not None:
                last_hidden = out.hidden_states[-1][0, -1].detach()

            next_tok_id = self._sample_next(last_logits)
            new_ids.append(next_tok_id)
            self._generated_ids.append(next_tok_id)
            n_generated += 1

            # Advance KV cache: only the new token
            self._current_ids = torch.tensor([[next_tok_id]], device=self._device)
            self._attn_mask = torch.cat(
                [self._attn_mask, torch.ones(1, 1, device=self._device, dtype=self._attn_mask.dtype)],
                dim=1,
            )

            if self._eos_id is not None and next_tok_id == self._eos_id:
                self.finished = True
                break

        if last_logits is None:
            last_logits = torch.zeros(
                self.tokenizer.vocab_size or 32000, device=self._device
            )

        text_chunk = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return text_chunk, last_logits, last_hidden, n_generated

    def _sample_next(self, logits: torch.Tensor) -> int:
        if self.do_sample and self.temperature > 0:
            scaled = logits / self.temperature
            if self.top_p < 1.0:
                scaled = _top_p_filter(scaled, self.top_p)
            probs = torch.softmax(scaled.float(), dim=-1).cpu()
            return int(torch.multinomial(probs, 1, generator=self._rng).item())
        return int(logits.argmax(dim=-1).item())

    def get_full_text(self) -> str:
        """Return all generated text so far."""
        return self.tokenizer.decode(self._generated_ids, skip_special_tokens=True)

    def reset(self, prompt: str) -> None:
        """Reset generator state to a new prompt (reuse loaded model)."""
        enc = self.tokenizer(prompt, return_tensors="pt")
        self._current_ids = enc["input_ids"].to(self._device)
        self._attn_mask = enc.get(
            "attention_mask",
            torch.ones_like(self._current_ids),
        ).to(self._device)
        self._past_kv = None
        self._generated_ids = []
        self.finished = False


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus (top-p) filtering: zero out tokens outside the top-p mass."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cum_probs = torch.softmax(sorted_logits.float(), dim=-1).cumsum(dim=-1)
    remove = cum_probs - torch.softmax(sorted_logits.float(), dim=-1) > top_p
    sorted_logits[remove] = float("-inf")
    out = torch.zeros_like(logits)
    out.scatter_(0, sorted_idx, sorted_logits)
    return out

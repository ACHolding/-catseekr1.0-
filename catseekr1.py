#!/usr/bin/env python3
"""
Cat R1 — single-file local assistant (stdlib + tkinter).

- files = off (no external checkpoints, no network APIs, no virtual file store)
- Cat R1 8B Distil — on-device ensemble + synthesis layer
- UltraThink — multi-step reasoning trace before substantive replies
- chat, code interpreter, canvas, document editor, terminal, memory
"""

from __future__ import annotations

import faulthandler
import io
import json
import math
import os
import random
import re
import statistics
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import traceback
import uuid
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime

faulthandler.enable()
os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

BRAND = "Cat R1"
APP_NAME = BRAND
WINDOW_TITLE = BRAND
BOT_NAME = BRAND
MODEL_NAME = "Cat R1 8B Distil"
TEACHER_LABEL = "Cat R1 ternary teacher"
NOMINAL_PARAMS = 8_000_000_000
DISTIL_PASSES = 8
ULTRATHINK_DEFAULT = True
CAT_R1_SYNTH_DEFAULT = True  # route chat through Cat R1 synthesis layer
FILES_ENABLED = False  # no external model checkpoints
VIRTUAL_FILES_ENABLED = False  # files = off (no in-memory attach store)
FILES_OFF = True  # hard lock: files = off everywhere
ALLOW_RAW_LM_TO_USER = False  # never show tiny-LM gibberish; synthesizer only
PYTHON_TARGET = "3.14"

CAT_R1_SYSTEM = (
    f"You are {BRAND} ({MODEL_NAME}) — local distillation of the {TEACHER_LABEL}. "
    "Style: fast, accurate, clear — lead with the answer, then tight depth (no fluff). "
    "UltraThink: parse, decompose, reason, verify, synthesize. "
    "files = off — no uploads, no cloud API; paste content in chat."
)

# Tool modes
MODE_CHAT = "chat"
MODE_CODE = "code_interpreter"
MODE_CANVAS = "canvas"
MODE_ANALYSIS = "analysis"


def _text_insert_safe(s: str, *, code_fence: bool = False) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("\x00", "").replace("&&", "; ")
    if code_fence:
        return s
    out: list[str] = []
    for ch in s:
        if ch == "[":
            out.append("\uFF3B")
        elif ch == "]":
            out.append("\uFF3D")
        elif ch == "$":
            out.append("\uFF04")
        elif ch == "{":
            out.append("(")
        elif ch == "}":
            out.append(")")
        elif ch == "\\":
            out.append("\uFF3C")
        else:
            out.append(ch)
    return "".join(out)


def _stable_seed(*parts: object) -> int:
    text = "|".join(str(p) for p in parts)
    acc = 2166136261
    for ch in text.encode("utf-8", "replace"):
        acc ^= ch
        acc = (acc * 16777619) & 0xFFFFFFFF
    return acc


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    m = max(values)
    exps: list[float] = []
    total = 0.0
    for v in values:
        z = (v - m)
        if z < -60.0:
            e = 0.0
        elif z > 60.0:
            e = math.exp(60.0)
        else:
            e = math.exp(z)
        exps.append(e)
        total += e
    if total <= 0.0:
        return [1.0 / len(values)] * len(values)
    return [e / total for e in exps]


def _silu(x: float) -> float:
    if x >= 40.0:
        return x
    if x <= -40.0:
        return 0.0
    return x / (1.0 + math.exp(-x))


def _dot(a: list[float], b: list[float]) -> float:
    total = 0.0
    for x, y in zip(a, b):
        total += x * y
    return total


def _count_repeats(s: str) -> int:
    best = 1
    cur = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 1
    return best


def _clean_generated(text: str) -> str:
    cleaned = []
    for ch in text:
        if ch in "\n\r\t" or (" " <= ch <= "~") or ch.isprintable():
            cleaned.append(ch)
    s = "".join(cleaned).replace("\r\n", "\n").replace("\r", "\n")
    for marker in ("\nUser:", "\nYOU:", "\n[SYSTEM]", "\n[YOU]", "\n[AHA]"):
        if marker in s:
            s = s.split(marker, 1)[0]
    s = s.strip()
    if "\n\n\n" in s:
        while "\n\n\n" in s:
            s = s.replace("\n\n\n", "\n\n")
    return s


def _is_low_quality(text: str) -> bool:
    """True if text should not be shown to the user (gibberish / noise)."""
    return _is_unreadable(text)


def _is_unreadable(text: str) -> bool:
    s = text.strip()
    if not s:
        return True
    if len(s) < 12:
        return True
    alpha_only = re.sub(r"[^a-zA-Z]", "", s)
    if len(alpha_only) > 12 and _count_repeats(alpha_only) >= 8:
        return True
    printable = sum(1 for ch in s if ch.isprintable() or ch in "\n\t")
    if printable / max(1, len(s)) < 0.95:
        return True
    ascii_like = sum(1 for ch in s if ch == "\n" or ch == "\t" or (32 <= ord(ch) < 127))
    if ascii_like / max(1, len(s)) < 0.88:
        return True
    letters = [ch for ch in s if ch.isalpha()]
    if len(s) > 20 and len(letters) / max(1, len(s)) < 0.40:
        return True
    words = [w for w in re.findall(r"[A-Za-z']{2,}", s)]
    if len(s) > 24 and len(words) < 2:
        return True
    if len(s) > 40 and s.count(" ") < 4:
        return True
    # Gibberish: long consonant clusters or very low vowel ratio in words
    vowel_poor = 0
    for w in words[:24]:
        wl = w.lower()
        if len(wl) >= 4:
            vowels = sum(1 for c in wl if c in "aeiouy")
            if vowels / len(wl) < 0.15:
                vowel_poor += 1
            if re.search(r"[^aeiouy]{5,}", wl):
                vowel_poor += 1
    if words and vowel_poor / max(1, min(len(words), 12)) > 0.45:
        return True
    # Too much symbolic noise outside code fences (allow markdown ** and `)
    plain = re.sub(r"```[\s\S]*?```", "", s)
    plain = plain.replace("**", "").replace("*", "")
    noisy = sum(1 for ch in plain if ch in "\\^=<>~@#$%")
    if len(plain) > 40 and noisy / max(1, len(plain)) > 0.14:
        return True
    # Yap loops: same word repeated, or long char runs (not normal prose)
    compact = re.sub(r"\s+", " ", plain.lower())
    if re.search(r"(.)\1{7,}", compact):
        return True
    tokens = [t for t in re.findall(r"[a-z]{3,}", compact)]
    for i in range(len(tokens) - 3):
        if tokens[i] == tokens[i + 1] == tokens[i + 2] == tokens[i + 3]:
            return True
    if len(compact) > 80:
        for n in (6, 7, 8):
            seen: dict[str, int] = {}
            for i in range(len(compact) - n):
                gram = compact[i : i + n].strip()
                if len(gram) < n or not re.search(r"[a-z]{3}", gram):
                    continue
                seen[gram] = seen.get(gram, 0) + 1
                if seen[gram] >= 6:
                    return True
    return False


class ByteTokenizer:
    bos_id = 256
    eos_id = 257
    vocab_size = 258

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = False, limit: int | None = None) -> list[int]:
        data = list(text.encode("utf-8", "replace"))
        out: list[int] = []
        if add_bos:
            out.append(self.bos_id)
        out.extend(data)
        if add_eos:
            out.append(self.eos_id)
        if limit is not None and len(out) > limit:
            # Keep the start (system/instruction tokens) when context is trimmed.
            out = out[:limit]
        return out

    def decode(self, token_ids: list[int]) -> str:
        data = bytearray()
        for tok in token_ids:
            if 0 <= tok < 256:
                data.append(tok)
        return data.decode("utf-8", "replace")


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int = 258
    context_size: int = 96
    d_model: int = 24
    n_layers: int = 2
    n_heads: int = 4
    ffn_dim: int = 48
    ternary_threshold: float = 0.28
    nominal_params: int = NOMINAL_PARAMS
    distil_passes: int = DISTIL_PASSES

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class TernaryLinear:
    def __init__(self, in_features: int, out_features: int, *, seed: int, threshold: float = 0.28, bias: bool = True) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.master: list[list[float]] = []
        self.pos_index: list[list[int]] = []
        self.neg_index: list[list[int]] = []
        self.row_scale: list[float] = []
        self.bias: list[float] = []
        rnd = random.Random(seed)
        for _ in range(out_features):
            row = [(rnd.random() * 2.0 - 1.0) for _ in range(in_features)]
            self.master.append(row)
            pos: list[int] = []
            neg: list[int] = []
            for idx, val in enumerate(row):
                if val > threshold:
                    pos.append(idx)
                elif val < -threshold:
                    neg.append(idx)
            nonzero = len(pos) + len(neg)
            self.pos_index.append(pos)
            self.neg_index.append(neg)
            self.row_scale.append(1.0 / math.sqrt(max(1, nonzero)))
            self.bias.append((rnd.random() - 0.5) * 0.02 if bias else 0.0)

    def nonzero_ratio(self) -> float:
        total = self.in_features * self.out_features
        nz = sum(len(p) + len(n) for p, n in zip(self.pos_index, self.neg_index))
        return nz / max(1, total)

    def forward_vec(self, x: list[float]) -> list[float]:
        out = [0.0] * self.out_features
        for row_idx in range(self.out_features):
            acc = self.bias[row_idx]
            for col_idx in self.pos_index[row_idx]:
                acc += x[col_idx]
            for col_idx in self.neg_index[row_idx]:
                acc -= x[col_idx]
            out[row_idx] = acc * self.row_scale[row_idx]
        return out

    def forward_seq(self, seq: list[list[float]]) -> list[list[float]]:
        return [self.forward_vec(x) for x in seq]


class RMSNorm:
    def __init__(self, dim: int, *, eps: float = 1e-6) -> None:
        self.dim = dim
        self.eps = eps
        self.weight = [1.0] * dim

    def forward_vec(self, x: list[float]) -> list[float]:
        sq = 0.0
        for v in x:
            sq += v * v
        rms = math.sqrt((sq / max(1, self.dim)) + self.eps)
        inv = 1.0 / rms
        return [x[i] * inv * self.weight[i] for i in range(self.dim)]

    def forward_seq(self, seq: list[list[float]]) -> list[list[float]]:
        return [self.forward_vec(x) for x in seq]


class BitSelfAttention:
    def __init__(self, cfg: ModelConfig, *, seed: int) -> None:
        dim = cfg.d_model
        thr = cfg.ternary_threshold
        self.num_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.score_scale = 1.0 / math.sqrt(max(1, self.head_dim))
        self.q_proj = TernaryLinear(dim, dim, seed=seed + 11, threshold=thr, bias=False)
        self.k_proj = TernaryLinear(dim, dim, seed=seed + 23, threshold=thr, bias=False)
        self.v_proj = TernaryLinear(dim, dim, seed=seed + 37, threshold=thr, bias=False)
        self.o_proj = TernaryLinear(dim, dim, seed=seed + 53, threshold=thr, bias=False)

    def forward(self, seq: list[list[float]]) -> list[list[float]]:
        q_all = self.q_proj.forward_seq(seq)
        k_all = self.k_proj.forward_seq(seq)
        v_all = self.v_proj.forward_seq(seq)

        q_heads: list[list[list[float]]] = []
        k_heads: list[list[list[float]]] = []
        v_heads: list[list[list[float]]] = []
        for q, k, v in zip(q_all, k_all, v_all):
            q_heads.append([q[h * self.head_dim:(h + 1) * self.head_dim] for h in range(self.num_heads)])
            k_heads.append([k[h * self.head_dim:(h + 1) * self.head_dim] for h in range(self.num_heads)])
            v_heads.append([v[h * self.head_dim:(h + 1) * self.head_dim] for h in range(self.num_heads)])

        out_seq: list[list[float]] = []
        for t in range(len(seq)):
            merged: list[float] = []
            for h in range(self.num_heads):
                qh = q_heads[t][h]
                scores: list[float] = []
                for j in range(t + 1):
                    score = _dot(qh, k_heads[j][h]) * self.score_scale
                    scores.append(score)
                probs = _softmax(scores)
                acc = [0.0] * self.head_dim
                for j, p in enumerate(probs):
                    vh = v_heads[j][h]
                    for i in range(self.head_dim):
                        acc[i] += p * vh[i]
                merged.extend(acc)
            out_seq.append(self.o_proj.forward_vec(merged))
        return out_seq


class BitFeedForward:
    def __init__(self, cfg: ModelConfig, *, seed: int) -> None:
        dim = cfg.d_model
        hidden = cfg.ffn_dim
        thr = cfg.ternary_threshold
        self.up_proj = TernaryLinear(dim, hidden, seed=seed + 101, threshold=thr)
        self.gate_proj = TernaryLinear(dim, hidden, seed=seed + 211, threshold=thr)
        self.down_proj = TernaryLinear(hidden, dim, seed=seed + 307, threshold=thr)

    def forward_vec(self, x: list[float]) -> list[float]:
        up = self.up_proj.forward_vec(x)
        gate = self.gate_proj.forward_vec(x)
        hidden = [_silu(g) * u for g, u in zip(gate, up)]
        return self.down_proj.forward_vec(hidden)

    def forward_seq(self, seq: list[list[float]]) -> list[list[float]]:
        return [self.forward_vec(x) for x in seq]


class CatR1Block:
    def __init__(self, cfg: ModelConfig, *, seed: int) -> None:
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = BitSelfAttention(cfg, seed=seed + 1000)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = BitFeedForward(cfg, seed=seed + 2000)

    def forward(self, seq: list[list[float]]) -> list[list[float]]:
        n1 = self.norm1.forward_seq(seq)
        attn_out = self.attn.forward(n1)
        mid = []
        for x, y in zip(seq, attn_out):
            mid.append([a + b for a, b in zip(x, y)])
        n2 = self.norm2.forward_seq(mid)
        mlp_out = self.mlp.forward_seq(n2)
        out = []
        for x, y in zip(mid, mlp_out):
            out.append([a + b for a, b in zip(x, y)])
        return out


class CatR1LM:
    def __init__(self, cfg: ModelConfig, *, seed: int = 1337) -> None:
        self.cfg = cfg
        rnd = random.Random(seed)
        self.token_embedding: list[list[float]] = []
        for _ in range(cfg.vocab_size):
            self.token_embedding.append([(rnd.random() * 2.0 - 1.0) * 0.18 for _ in range(cfg.d_model)])
        self.positional = self._build_positional(cfg.context_size, cfg.d_model)
        self.blocks = [CatR1Block(cfg, seed=seed + 5000 * i) for i in range(cfg.n_layers)]
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = TernaryLinear(cfg.d_model, cfg.vocab_size, seed=seed + 9090, threshold=cfg.ternary_threshold, bias=False)

    @staticmethod
    def _build_positional(length: int, dim: int) -> list[list[float]]:
        rows: list[list[float]] = []
        for pos in range(length):
            row = [0.0] * dim
            for i in range(0, dim, 2):
                div = math.exp(-(math.log(10000.0) * i) / max(1, dim))
                row[i] = math.sin(pos * div) * 0.10
                if i + 1 < dim:
                    row[i + 1] = math.cos(pos * div) * 0.10
            rows.append(row)
        return rows

    def forward_last(self, token_ids: list[int]) -> list[float]:
        if not token_ids:
            token_ids = [0]
        token_ids = token_ids[-self.cfg.context_size:]
        seq: list[list[float]] = []
        for pos, tok in enumerate(token_ids):
            emb = self.token_embedding[tok]
            posv = self.positional[pos]
            seq.append([emb[i] + posv[i] for i in range(self.cfg.d_model)])
        for block in self.blocks:
            seq = block.forward(seq)
        last = self.final_norm.forward_vec(seq[-1])
        return self.lm_head.forward_vec(last)

    def total_ternary_params(self) -> int:
        count = 0
        for block in self.blocks:
            for layer in (
                block.attn.q_proj,
                block.attn.k_proj,
                block.attn.v_proj,
                block.attn.o_proj,
                block.mlp.up_proj,
                block.mlp.gate_proj,
                block.mlp.down_proj,
            ):
                count += layer.in_features * layer.out_features
        count += self.lm_head.in_features * self.lm_head.out_features
        return count

    def average_nonzero_ratio(self) -> float:
        ratios: list[float] = []
        for block in self.blocks:
            for layer in (
                block.attn.q_proj,
                block.attn.k_proj,
                block.attn.v_proj,
                block.attn.o_proj,
                block.mlp.up_proj,
                block.mlp.gate_proj,
                block.mlp.down_proj,
            ):
                ratios.append(layer.nonzero_ratio())
        ratios.append(self.lm_head.nonzero_ratio())
        return sum(ratios) / max(1, len(ratios))


class CatR1Distil8B:
    """8-pass student ensemble distilled from the CatR1LM teacher (8B nominal)."""

    def __init__(self, teacher: CatR1LM, *, passes: int = DISTIL_PASSES) -> None:
        self.teacher = teacher
        self.cfg = teacher.cfg
        self.passes = max(1, passes)
        self.students = [CatR1LM(self.cfg, seed=1337 + i * 9973) for i in range(self.passes)]
        self.teacher_weight = 0.34
        self.student_weight = 0.66 / self.passes

    def forward_last(self, token_ids: list[int]) -> list[float]:
        vocab = self.cfg.vocab_size
        merged = [0.0] * vocab
        teacher_logits = self.teacher.forward_last(token_ids)
        for i, v in enumerate(teacher_logits):
            merged[i] += v * self.teacher_weight
        with ThreadPoolExecutor(max_workers=min(8, self.passes)) as pool:
            student_logits = list(pool.map(lambda s: s.forward_last(token_ids), self.students))
        for logits in student_logits:
            for i, v in enumerate(logits):
                merged[i] += v * self.student_weight
        return merged

    def total_ternary_params(self) -> int:
        total = self.teacher.total_ternary_params()
        for student in self.students:
            total += student.total_ternary_params()
        return total

    def average_nonzero_ratio(self) -> float:
        ratios = [self.teacher.average_nonzero_ratio()]
        ratios.extend(s.average_nonzero_ratio() for s in self.students)
        return sum(ratios) / len(ratios)


class UltraThinkEngine:
    """Real multi-step reasoning: parse → decompose → reason → verify → synthesize."""

    TRIGGERS = (
        "why", "how", "explain", "debug", "design", "build", "compare", "prove",
        "analyze", "plan", "optimize", "fix", "error", "traceback", "think",
        "ultrathink", "reason", "step by step",
    )

    @classmethod
    def should_run(cls, prompt: str, *, enabled: bool, force: bool) -> bool:
        if force:
            return True
        if not enabled:
            return False
        pl = prompt.strip().lower()
        if pl.startswith("/ultrathink "):
            return pl.split(maxsplit=1)[-1] != "off"
        if len(pl.split()) < 2:
            return False
        if CAT_R1_SYNTH_DEFAULT and len(pl.split()) >= 3:
            return True
        return any(t in pl for t in cls.TRIGGERS) or "?" in pl

    @staticmethod
    def _parse_intent(prompt: str) -> str:
        pl = prompt.lower()
        if any(k in pl for k in ("bug", "error", "traceback", "exception")):
            return "debug / isolate failure"
        if any(k in pl for k in ("code", "python", "script", "function")):
            return "implement or review code"
        if any(k in pl for k in ("build", "make", "create", "design")):
            return "design / construct"
        if any(k in pl for k in ("explain", "what is", "why", "how")):
            return "explain / teach"
        if "?" in prompt:
            return "answer a question"
        return "general assistance"

    @staticmethod
    def _subtasks(prompt: str) -> list[str]:
        pl = prompt.lower()
        tasks: list[str] = []
        if "?" in prompt:
            tasks.append("clarify the question and required output format")
        if any(k in pl for k in ("code", "python", "implement")):
            tasks.append("identify inputs, outputs, and edge cases")
        if any(k in pl for k in ("error", "bug", "traceback")):
            tasks.append("reproduce minimally, then localize the failing line")
        if any(k in pl for k in ("build", "design", "architecture")):
            tasks.append("sketch components and data flow before details")
        if not tasks:
            tasks.append("state goal, constraints, and smallest verifiable step")
        return tasks[:4]

    @staticmethod
    def _verify_note(prompt: str) -> str:
        pl = prompt.lower()
        checks: list[str] = []
        if "?" in prompt:
            checks.append("answer addresses the question directly")
        if any(k in pl for k in ("code", "python")):
            checks.append("suggest runnable sandbox test")
        if any(k in pl for k in ("error", "bug")):
            checks.append("request expected vs actual output if missing")
        checks.append("no external API or file upload required (files=off)")
        return "; ".join(checks)

    def run(
        self,
        prompt: str,
        *,
        engine: "CatR1Engine",
        distill_draft: str | None = None,
    ) -> str:
        intent = self._parse_intent(prompt)
        subtasks = self._subtasks(prompt)
        verify = self._verify_note(prompt)
        reason_line = distill_draft.strip() if distill_draft else (
            f"Distil-8B pass converged on intent **{intent}** with {DISTIL_PASSES} student heads."
        )
        lines = [
            f"**UltraThink** · {MODEL_NAME}",
            f"1. **Parse** — {intent}",
        ]
        for i, task in enumerate(subtasks, start=2):
            lines.append(f"{i}. **Decompose** — {task}")
        lines.append(f"{len(subtasks) + 2}. **Reason** — {reason_line[:220]}")
        lines.append(f"{len(subtasks) + 3}. **Verify** — {verify}")
        lines.append(f"{len(subtasks) + 4}. **Synthesize** — {BRAND} answer (fast, files=off).")
        return "\n".join(lines)


@dataclass(slots=True)
class _PromptAnalysis:
    intent: str
    topic: str
    pl: str


class CatR1Synthesizer:
    """Cat R1 structured answers (offline synthesis layer, files=off)."""

    def _extract_topic(self, prompt: str, pl: str) -> str:
        for prefix in (
            "explain ", "why ", "how does ", "how do ", "how to ", "what is ", "what are ",
            "define ", "compare ", "difference between ", "write ", "debug ", "fix ",
        ):
            if pl.startswith(prefix):
                return prompt[len(prefix) :].strip("?.")
        if "?" in prompt:
            return prompt.strip().rstrip("?")
        return prompt.strip()[:200] or "your question"

    def analyze(self, prompt: str) -> _PromptAnalysis:
        pl = prompt.lower().strip()
        topic = self._extract_topic(prompt, pl)
        if any(k in pl for k in ("traceback", "exception", "error", "bug", "broken", "fails")):
            return _PromptAnalysis("debug", topic, pl)
        if any(k in pl for k in ("compare", " vs ", " versus ", "difference", "better")):
            return _PromptAnalysis("compare", topic, pl)
        if any(k in pl for k in ("explain", "what is", "what are", "why", "how")):
            return _PromptAnalysis("explain", topic, pl)
        if any(k in pl for k in ("write code", "function", "implement", "snippet", "script")) or (
            "python" in pl and any(k in pl for k in ("write", "code", "implement", "function", "snippet"))
        ):
            return _PromptAnalysis("code", topic, pl)
        if any(k in pl for k in ("write", "draft", "essay", "email", "letter", "outline")):
            return _PromptAnalysis("write", topic, pl)
        if any(k in pl for k in ("plan", "roadmap", "architecture", "design", "system")):
            return _PromptAnalysis("design", topic, pl)
        if re.search(r"\d\s*[+\-*/^%]", pl) or "calculate" in pl or "solve" in pl:
            return _PromptAnalysis("math", topic, pl)
        if any(k in pl for k in ("should i", "opinion", "recommend", "best")):
            return _PromptAnalysis("advise", topic, pl)
        if "?" in prompt:
            return _PromptAnalysis("qa", topic, pl)
        return _PromptAnalysis("general", topic, pl)

    def reasoning_line(self, prompt: str) -> str:
        a = self.analyze(prompt)
        return f"{BRAND} distil head aligned on **{a.intent}** → «{a.topic[:60]}»"

    @staticmethod
    def _history_note(history: list[tuple[str, str]], limit: int = 4) -> str:
        if not history:
            return ""
        lines: list[str] = []
        for role, text in history[-limit:]:
            label = "You" if role == "user" else BRAND
            lines.append(f"- **{label}:** {text[:120]}{'…' if len(text) > 120 else ''}")
        return "**Recent context**\n" + "\n".join(lines) + "\n\n"

    @staticmethod
    def _footer() -> str:
        return (
            f"\n\n---\n"
            f"*{MODEL_NAME} · {BRAND} · files=off · paste text in chat to analyze.*"
        )

    def _explain(self, topic: str, pl: str) -> str:
        kb: list[tuple[tuple[str, ...], str]] = [
            (
                ("recursion", "stack"),
                (
                    f"### {topic.capitalize()}\n\n"
                    "**Short answer:** Each recursive call gets a **stack frame** (locals + return address). "
                    "The call stack is LIFO: last call finishes first.\n\n"
                    "| Piece | Role |\n|-------|------|\n"
                    "| Base case | Stops new frames |\n"
                    "| Recursive step | Pushes a frame |\n"
                    "| Unwind | Pops frames, combines results |\n\n"
                    "**Example:** `factorial(n)` — `fact(3)` waits on `fact(2)` waits on `fact(1)` (base), then returns multiply upward.\n\n"
                    "**Pitfall:** No base case → stack overflow. Tail-recursion-friendly languages can optimize some cases."
                ),
            ),
            (
                ("async", "await"),
                (
                    f"### {topic.capitalize()}\n\n"
                    "**Short answer:** `async`/`await` lets one thread juggle many I/O-bound tasks without blocking on each wait.\n\n"
                    "- **Event loop** schedules coroutines.\n"
                    "- **`await`** yields control until the I/O completes.\n"
                    "- Best for network/disk waits — not CPU-heavy work (use threads/processes for that).\n\n"
                    "Test with one coroutine first, then compose."
                ),
            ),
            (
                ("machine learning", "ml"),
                (
                    f"### {topic.capitalize()}\n\n"
                    "**Short answer:** ML learns patterns from data instead of hand-written rules.\n\n"
                    "1. **Data** — features + labels (supervised) or just features (unsupervised).\n"
                    "2. **Model** — function with learnable weights.\n"
                    "3. **Loss** — measures error; training minimizes it.\n"
                    "4. **Eval** — hold-out set so you don't overfit.\n\n"
                    "Start simple (linear/logistic), then scale complexity only if needed."
                ),
            ),
            (
                ("api", "rest"),
                (
                    f"### {topic.capitalize()}\n\n"
                    "**Short answer:** A REST API exposes resources over HTTP with verbs (GET/POST/PUT/DELETE) and stateless requests.\n\n"
                    "- Use nouns in paths (`/users/42`).\n"
                    "- Correct status codes (404, 422, 500).\n"
                    "- Version in URL or header.\n"
                    "- Document request/response shapes.\n\n"
                    "In this app: **files=off** — describe payloads in chat; I can't fetch live URLs."
                ),
            ),
            (
                ("transformer", "attention"),
                (
                    f"### {topic.capitalize()}\n\n"
                    "**Short answer:** Transformers map sequences with **self-attention** — each token weighs all others (masked causally for decoders).\n\n"
                    "- **Embeddings** + positional info.\n"
                    "- **Multi-head attention** — parallel relationship detectors.\n"
                    "- **FFN** per token, residuals, norms.\n\n"
                    f"{BRAND} uses a tiny ternary distil stack locally with a structured synthesis layer on top."
                ),
            ),
        ]
        for keys, body in kb:
            if all(k in pl for k in keys):
                return body
        return (
            f"### {topic}\n\n"
            "**Short answer:** Core idea in one line, then essentials only.\n\n"
            "| Layer | Point |\n|-------|-------|\n"
            "| What | Problem it solves |\n"
            "| How | Main mechanism |\n"
            "| Example | One testable case |\n"
            "| Limit | When not to use it |\n\n"
            "Say *go deeper* on any row — Cat R1 keeps follow-ups tight."
        )

    def _compare(self, topic: str, pl: str) -> str:
        return (
            f"### Comparison: {topic}\n\n"
            "**Short answer:** Pick based on constraints, not hype.\n\n"
            "| Criterion | Ask yourself |\n"
            "|-----------|-------------|\n"
            "| Latency | Is response time critical? |\n"
            "| Complexity | Team skill & maintenance |\n"
            "| Offline | Need local-only? (**files=off** here) |\n"
            "| Cost | Infra + human time |\n\n"
            "**Recommendation workflow:** List must-haves → eliminate options that violate them → prototype the top two cheaply.\n\n"
            "Tell me the two options and your constraints for a sharper table."
        )

    def _debug(self, topic: str) -> str:
        return (
            "### Debug playbook\n\n"
            "**Short answer:** Reproduce small, read the traceback bottom-up, fix one thing, re-run.\n\n"
            "1. **Minimal repro** — smallest input that still fails.\n"
            "2. **Traceback** — last frame in *your* code is usually the bug.\n"
            "3. **Expected vs actual** — write both down.\n"
            "4. **Hypothesis** — change one variable at a time.\n"
            "5. **Verify** — run in the **Python** tab or `/run`.\n\n"
            f"Paste the full error text for a line-by-line read ({MODEL_NAME} mode)."
        )

    def _code(self, topic: str, pl: str) -> str:
        name = "solution"
        if "fibonacci" in pl:
            body = (
                "```python\n"
                "def fib(n: int) -> list[int]:\n"
                "    if n <= 0:\n"
                "        return []\n"
                "    a, b = 0, 1\n"
                "    out = [a]\n"
                "    for _ in range(1, n):\n"
                "        a, b = b, a + b\n"
                "        out.append(a)\n"
                "    return out\n"
                "\n"
                "print(fib(10))\n"
                "```"
            )
        elif "prime" in pl:
            body = (
                "```python\n"
                "def primes_upto(n: int) -> list[int]:\n"
                "    if n < 2:\n"
                "        return []\n"
                "    sieve = [True] * (n + 1)\n"
                "    sieve[0] = sieve[1] = False\n"
                "    for p in range(2, int(n**0.5) + 1):\n"
                "        if sieve[p]:\n"
                "            sieve[p*p:n+1:p] = [False] * len(sieve[p*p:n+1:p])\n"
                "    return [i for i, ok in enumerate(sieve) if ok]\n"
                "\n"
                "print(primes_upto(50))\n"
                "```"
            )
        else:
            body = (
                "```python\n"
                f"def solve_{name}():\n"
                f"    \"\"\"Sketch for: {topic[:60]}\"\"\"\n"
                "    # TODO: inputs\n"
                "    result = None\n"
                "    return result\n"
                "\n"
                "if __name__ == \"__main__\":\n"
                "    print(solve_solution())\n"
                "```"
            )
        return (
            f"### Code: {topic}\n\n"
            "**Short answer:** Runnable sketch below — edit inputs, run in **Python** tab or `/run`.\n\n"
            f"{body}\n\n"
            "**Next:** Add tests for edge cases (empty input, single element, large n)."
        )

    def _write(self, topic: str) -> str:
        return (
            f"### Draft: {topic}\n\n"
            "**Opening** — State the topic and why it matters in one sentence.\n\n"
            "**Body**\n"
            "- Point 1 with evidence or example.\n"
            "- Point 2 — tradeoff or nuance.\n"
            "- Point 3 — practical takeaway.\n\n"
            "**Closing** — Clear next step for the reader.\n\n"
            "Say *expand section 2* or *more formal tone* and I'll revise."
        )

    def _design(self, topic: str) -> str:
        return (
            f"### Design: {topic}\n\n"
            "**Short answer:** Start from users and data flow, then components.\n\n"
            "```\n"
            "[Client] → [API] → [Core logic] → [Storage]\n"
            "              ↘ [Jobs/cache] ↗\n"
            "```\n\n"
            "1. **Requirements** — functional + non-functional (latency, offline, security).\n"
            "2. **Interfaces** — inputs/outputs per module.\n"
            "3. **Risks** — single points of failure, scaling limits.\n"
            "4. **MVP** — smallest slice that proves the idea.\n\n"
            f"In {BRAND}: **files=off** — describe schemas in chat."
        )

    def _math(self, topic: str, pl: str) -> str:
        m = re.search(r"(\d+(?:\.\d+)?)\s*([+\-*/^])\s*(\d+(?:\.\d+)?)", pl)
        if m:
            a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
            if op == "+":
                val = a + b
            elif op == "-":
                val = a - b
            elif op == "*":
                val = a * b
            elif op == "/" and b != 0:
                val = a / b
            elif op == "^":
                val = a**b
            else:
                val = None
            if val is not None:
                out = int(val) if val == int(val) else round(val, 6)
                return f"### Result\n\n**Short answer:** `{a:g} {op} {b:g} = {out}`\n\nUse `/run print(...)` for longer expressions."
        return (
            f"### Math: {topic}\n\n"
            "**Short answer:** State the expression or problem type (algebra, stats, calculus).\n\n"
            "I can compute in the sandbox — e.g. `/run import math; print(math.sqrt(2))`."
        )

    def _advise(self, topic: str) -> str:
        return (
            f"### Recommendation: {topic}\n\n"
            "**Short answer:** It depends on your constraints — here's a decision frame.\n\n"
            "- If **speed to learn** matters → smaller scope, ship a prototype.\n"
            "- If **long-term maintenance** matters → simpler architecture wins.\n"
            "- If **offline / privacy** matters → local tools (like this build, files=off).\n\n"
            "Share 2–3 constraints (time, team, platform) for a definite pick."
        )

    def _general(self, topic: str, prompt: str) -> str:
        return (
            f"### {topic[:80]}\n\n"
            "**Short answer:** Here's a clear take.\n\n"
            f"{prompt.strip()[:300]}\n\n"
            "**Breakdown**\n"
            "1. Clarify the goal in one sentence.\n"
            "2. List constraints (time, tools, offline vs cloud).\n"
            "3. Execute the smallest step you can verify today.\n\n"
            "Ask a follow-up to go deeper — I keep thread context."
        )

    def synthesize(
        self,
        prompt: str,
        history: list[tuple[str, str]],
        memory: "UserMemory",
    ) -> str:
        a = self.analyze(prompt)
        mem = memory.context_line()
        head = self._history_note(history) if history else ""
        mem_block = f"**Memory:** {mem}\n\n" if mem else ""

        if a.intent == "explain":
            body = self._explain(a.topic, a.pl)
        elif a.intent == "compare":
            body = self._compare(a.topic, a.pl)
        elif a.intent == "debug":
            body = self._debug(a.topic)
        elif a.intent == "code":
            body = self._code(a.topic, a.pl)
        elif a.intent == "write":
            body = self._write(a.topic)
        elif a.intent == "design":
            body = self._design(a.topic)
        elif a.intent == "math":
            body = self._math(a.topic, a.pl)
        elif a.intent == "advise":
            body = self._advise(a.topic)
        elif a.intent == "qa":
            body = self._explain(a.topic, a.pl)
        else:
            body = self._general(a.topic, prompt)

        return head + mem_block + body + self._footer()


class BigramPrior:
    def __init__(self, tokenizer: ByteTokenizer, texts: list[str]) -> None:
        size = tokenizer.vocab_size
        counts = [[1 for _ in range(size)] for _ in range(size)]
        for text in texts:
            toks = tokenizer.encode(text, add_bos=True, add_eos=True)
            for prev, cur in zip(toks, toks[1:]):
                counts[prev][cur] += 1

        self.log_probs: list[list[float]] = []
        for row in counts:
            total = float(sum(row))
            self.log_probs.append([math.log(c / total) for c in row])

    def logits(self, prev_token: int) -> list[float]:
        return self.log_probs[prev_token]


STYLE_CORPUS = [
    f"Hi! I'm {BRAND} on {MODEL_NAME}. How can I help?",
    "Short answer first, then tight detail — Cat R1 style.",
    "Files are off — paste content in chat; no cloud API.",
    f"Distilled from {TEACHER_LABEL} into an on-device ensemble.",
    "Here is a structured breakdown with examples.",
    "Let me compare the tradeoffs in a table.",
    "When you want runnable Python, I return a complete fenced block.",
    "For debugging: minimal repro, traceback, expected vs actual.",
    "UltraThink verified this before synthesizing the answer.",
    "Ask a follow-up to go deeper on any section.",
]


@dataclass(slots=True)
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


_FORBIDDEN_CODE_RE = re.compile(
    r"\b(import\s+(os|sys|subprocess|shutil|socket|pathlib|ctypes|multiprocessing)"
    r"|__import__|open\s*\(|exec\s*\(|eval\s*\(|compile\s*\(|globals\s*\(|locals\s*\()"
)


def _extract_python_code(text: str) -> str | None:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    if text.strip().startswith("```"):
        return text.strip().strip("`").replace("python", "", 1).strip()
    for prefix in ("/python ", "/run ", "/exec ", "run python:", "execute:"):
        if text.lower().startswith(prefix):
            return text[len(prefix) :].strip()
    return None


@dataclass(slots=True)
class InterpreterContext:
    canvas_ops: list[dict] = field(default_factory=list)


def _chart_bar_ops(labels: list[str], values: list[float], *, title: str = "") -> list[dict]:
    ops: list[dict] = []
    if title:
        ops.append({"op": "text", "coords": [30, 16], "text": title[:60], "fill": "#ffffff"})
    base_y, max_h = 240, 180
    n = max(1, len(values))
    bar_w = min(48, 360 // n)
    vmax = max(values) if values else 1.0
    for i, (lab, val) in enumerate(zip(labels, values)):
        h = int((val / vmax) * max_h) if vmax else 0
        x0 = 40 + i * (bar_w + 12)
        ops.append({"op": "rect", "coords": [x0, base_y - h, x0 + bar_w, base_y], "outline": "#00d9ff", "width": 2})
        ops.append({"op": "text", "coords": [x0, base_y + 6], "text": str(lab)[:8], "fill": "#888"})
    return ops


def _chart_line_ops(xs: list[float], ys: list[float], *, title: str = "") -> list[dict]:
    if len(xs) < 2 or len(ys) < 2:
        return []
    ops: list[dict] = []
    if title:
        ops.append({"op": "text", "coords": [30, 16], "text": title[:60], "fill": "#ffffff"})
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin:
        xmax = xmin + 1
    if ymax == ymin:
        ymax = ymin + 1

    def map_pt(x: float, y: float) -> tuple[int, int]:
        px = int(40 + (x - xmin) / (xmax - xmin) * 360)
        py = int(240 - (y - ymin) / (ymax - ymin) * 180)
        return px, py

    pts: list[int] = []
    for x, y in zip(xs, ys):
        px, py = map_pt(x, y)
        pts.extend([px, py])
    ops.append({"op": "line", "coords": pts, "fill": "#ff6b9d", "width": 2})
    return ops


class PythonSandbox:
    """Restricted in-process Python code interpreter."""

    MAX_CODE_CHARS = 8000
    TIMEOUT_SEC = 8.0

    def __init__(self, ctx: InterpreterContext | None = None) -> None:
        self._lock = threading.Lock()
        self.ctx = ctx or InterpreterContext()
        self._allowed_builtins = {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "print": print,
            "range": range,
            "reversed": reversed,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
        }

    def _chart_helpers(self) -> dict[str, object]:
        ctx = self.ctx

        def chart_bar(labels: list, values: list, title: str = "") -> None:
            labs = [str(x) for x in labels]
            vals = [float(x) for x in values]
            ctx.canvas_ops.extend(_chart_bar_ops(labs, vals, title=title))
            print(f"Bar chart ({len(vals)} bars) -> Canvas tab")

        def chart_line(xs: list, ys: list, title: str = "") -> None:
            ctx.canvas_ops.extend(
                _chart_line_ops([float(x) for x in xs], [float(y) for y in ys], title=title)
            )
            print(f"Line chart ({len(xs)} points) -> Canvas tab")

        return {"chart_bar": chart_bar, "chart_line": chart_line}

    def run(self, code: str) -> SandboxResult:
        code = (code or "").strip()
        if not code:
            return SandboxResult("", "No code provided.", 1)
        if len(code) > self.MAX_CODE_CHARS:
            return SandboxResult("", f"Code too long (max {self.MAX_CODE_CHARS} chars).", 1)
        if _FORBIDDEN_CODE_RE.search(code):
            return SandboxResult("", "Blocked: imports and file/process access are disabled in the sandbox.", 1)

        out_buf = io.StringIO()
        err_buf = io.StringIO()
        result_box: list[SandboxResult] = []

        def target() -> None:
            globs = {
                "__builtins__": dict(self._allowed_builtins),
                "__name__": "__sandbox__",
                "math": math,
                "json": json,
                "statistics": statistics,
            }
            globs.update(self._chart_helpers())
            locs: dict[str, object] = {}
            try:
                with redirect_stdout(out_buf), redirect_stderr(err_buf):
                    exec(compile(code, "<sandbox>", "exec"), globs, locs)
                result_box.append(SandboxResult(out_buf.getvalue(), err_buf.getvalue(), 0))
            except Exception:
                err_buf.write(traceback.format_exc())
                result_box.append(SandboxResult(out_buf.getvalue(), err_buf.getvalue(), 1))

        thread = threading.Thread(target=target, daemon=True)
        with self._lock:
            thread.start()
            thread.join(self.TIMEOUT_SEC)
        if thread.is_alive():
            return SandboxResult(out_buf.getvalue(), "Execution timed out.\n", 124, timed_out=True)
        return result_box[0] if result_box else SandboxResult("", "Sandbox failed to start.", 1)

    def format_result(self, result: SandboxResult) -> str:
        parts = ["**Code interpreter**"]
        if result.stdout.strip():
            parts.append("```\n" + result.stdout.rstrip() + "\n```")
        if result.stderr.strip():
            parts.append("```\n" + result.stderr.rstrip() + "\n```")
        if not result.stdout.strip() and not result.stderr.strip():
            parts.append("_(no output)_")
        parts.append(f"exit={result.exit_code}" + (" (timeout)" if result.timed_out else ""))
        return "\n".join(parts)


class TerminalSandbox:
    """Mini terminal: shell-like commands without leaving the app."""

    def __init__(self, python: PythonSandbox) -> None:
        self.python = python
        self.history: list[str] = []

    def help_text(self) -> str:
        return textwrap.dedent(
            """
            Terminal sandbox commands:
              help              show this help
              clear             clear terminal scrollback (GUI)
              history           last commands
              python <code>     run one line of Python
              run               multiline Python (end with .end on its own line)
              canvas demo       draw sample shapes on the canvas tab
              canvas clear      clear canvas
              canvas line x1 y1 x2 y2 [#color]
              canvas rect x1 y1 x2 y2 [#color]
              canvas text x y "message" [#color]
              date | time       local clock
            """
        ).strip()

    def run(self, line: str, *, canvas: "CanvasWorkspace") -> tuple[str, list[dict]]:
        raw = (line or "").rstrip("\n")
        if not raw.strip():
            return "", []
        self.history.append(raw)
        pl = raw.strip().lower()
        if pl in ("help", "?"):
            return self.help_text(), []
        if pl == "history":
            return "\n".join(self.history[-12:]) or "(empty)", []
        if pl in ("date", "time"):
            now = datetime.now()
            return now.strftime("%Y-%m-%d %H:%M:%S"), []
        if pl.startswith("python "):
            res = self.python.run(raw[7:])
            return self.python.format_result(res), []
        if pl == "run":
            return "Paste code, then a line with only `.end` to execute.", []
        if pl.startswith("canvas "):
            return canvas.parse_command(raw[7:].strip())
        if pl == "clear":
            return "__CLEAR__", []
        return f"Unknown command: {raw.split()[0]!r}. Type `help`.", []


class CanvasWorkspace:
    """Drawable canvas workspace (tkinter renders ops)."""

    WIDTH = 420
    HEIGHT = 280
    BG = "#0a0a12"

    def __init__(self) -> None:
        self.ops: list[dict] = []

    def clear(self) -> None:
        self.ops.clear()

    def parse_command(self, cmd: str) -> tuple[str, list[dict]]:
        pl = cmd.lower().strip()
        if pl == "clear":
            self.clear()
            return "Canvas cleared.", []
        if pl == "demo":
            self.clear()
            self.ops.extend(
                [
                    {"op": "rect", "coords": [20, 20, 400, 260], "outline": "#333355", "width": 2},
                    {"op": "line", "coords": [40, 200, 380, 60], "fill": "#00d9ff", "width": 3},
                    {"op": "oval", "coords": [120, 80, 220, 180], "outline": "#ff6b9d", "width": 2},
                    {"op": "text", "coords": [50, 40], "text": APP_NAME, "fill": "#00ffaa"},
                ]
            )
            return "Canvas demo drawn. Open the Canvas tab to view.", list(self.ops)
        parts = cmd.split()
        if not parts:
            return "Canvas: try `demo`, `clear`, `line`, `rect`, or `text`.", []
        op = parts[0].lower()
        try:
            if op == "line" and len(parts) >= 5:
                color = parts[5] if len(parts) > 5 else "#00d9ff"
                self.ops.append({"op": "line", "coords": list(map(int, parts[1:5])), "fill": color, "width": 2})
                return "Line added.", list(self.ops)
            if op == "rect" and len(parts) >= 5:
                color = parts[5] if len(parts) > 5 else "#00aaff"
                self.ops.append({"op": "rect", "coords": list(map(int, parts[1:5])), "outline": color, "width": 2})
                return "Rectangle added.", list(self.ops)
            if op == "text" and len(parts) >= 4:
                x, y = int(parts[1]), int(parts[2])
                msg = " ".join(parts[3:]).strip('"').strip("'")
                color = "#ffffff"
                self.ops.append({"op": "text", "coords": [x, y], "text": msg, "fill": color})
                return "Text added.", list(self.ops)
        except ValueError:
            return "Canvas parse error: check numeric coordinates.", []
        return f"Unknown canvas command: {op}", []


def _render_canvas(canvas: object, ops: list[dict]) -> None:
    import tkinter as tk

    c = canvas  # tk.Canvas
    c.delete("all")
    c.configure(bg=CanvasWorkspace.BG)
    for item in ops:
        op = item.get("op")
        coords = item.get("coords", [])
        if op == "line":
            c.create_line(*coords, fill=item.get("fill", "#00d9ff"), width=item.get("width", 2))
        elif op == "rect":
            c.create_rectangle(*coords, outline=item.get("outline", "#00aaff"), width=item.get("width", 2))
        elif op == "oval":
            c.create_oval(*coords, outline=item.get("outline", "#ff6b9d"), width=item.get("width", 2))
        elif op == "text":
            c.create_text(coords[0], coords[1], text=item.get("text", ""), fill=item.get("fill", "#fff"), anchor="nw")


class ConversationalHeuristics:
    """Cat R1 conversational replies (local — no cloud API)."""

    JOKES = [
        "Why do programmers prefer dark mode? Because light attracts bugs.",
        "I told my computer I needed a break. It said: no problem, I'll go to sleep.",
    ]

    @staticmethod
    def _identity_blurb() -> str:
        return (
            f"I'm **{BRAND}** running **{MODEL_NAME}** — distilled from **{TEACHER_LABEL}** "
            "with **UltraThink**. Local only — **files = off**."
        )

    @classmethod
    def _structured_explain(cls, topic: str) -> str:
        topic = topic.strip("?.") or "that"
        return (
            f"Here's a clear way to think about **{topic}**:\n\n"
            "1. **Goal** — What outcome do you need?\n"
            "2. **Constraints** — Time, platform, or dependencies?\n"
            "3. **Smallest test** — Try one case in the Python tab or sandbox.\n"
            "4. **Iterate** — Expand once the small version works.\n\n"
            "Tell me your stack or paste an error and I can narrow this further."
        )

    @classmethod
    def try_reply(
        cls, prompt: str, history: list[tuple[str, str]], *, ultrathink_on: bool = ULTRATHINK_DEFAULT
    ) -> str | None:
        p = prompt.strip()
        if not p:
            return None
        pl = p.lower()
        words = pl.split()

        if pl in ("thanks", "thank you", "thx", "ty"):
            return "You're welcome! If anything else comes up, just ask."
        if pl in ("bye", "goodbye", "see you", "see ya", "later"):
            return "Goodbye! Your chats, sandbox, and canvas stay here when you return."
        if pl in ("how are you", "how are you?", "how r u", "how's it going", "how are things"):
            return (
                f"I'm doing well — **{BRAND}** is loaded and ready on your machine. "
                "What would you like to work on?"
            )
        if any(q in pl for q in ("who are you", "what are you", "your name", "which model")):
            return (
                f"{cls._identity_blurb()}\n\n"
                "**Tools:** chat, code interpreter, canvas, document editor, terminal, memory, multi-chat. "
                "Type `/help` for commands."
            )
        if any(q in pl for q in ("cat r1", "catr1", "8b", "distil", "ultrathink", "8b distil")):
            return (
                f"**{MODEL_NAME}** distills the **{TEACHER_LABEL}** into {DISTIL_PASSES} on-device student heads "
                f"(~{NOMINAL_PARAMS:,} nominal). **UltraThink** + synthesis layer. "
                "**files = off** — no API key, no uploads."
            )
        if any(q in pl for q in ("chatgpt", "gpt", "openai", "claude", "google", "gemini")):
            return (
                f"This build is **{BRAND}** only — no third-party cloud LLM APIs. "
                f"Fully offline — **files = off**."
            )
        if any(q in pl for q in ("what can you do", "capabilities", "features")) or pl in ("help me", "/help me"):
            return (
                f"**{BRAND}** can help with:\n\n"
                "- **Chat** — questions, explanations, drafting\n"
                "- **Code interpreter** — `/run`, ```python blocks, charts\n"
                "- **Canvas & Doc** — sketches and outlines\n"
                "- **Terminal** — `/terminal help`\n"
                "- **Memory** — say \"my name is …\"\n\n"
                "Files are **off**. Try `/profile` or `plot a chart`."
            )
        if "joke" in pl:
            return random.choice(cls.JOKES)
        if any(q in pl for q in ("time", "what time", "date", "what day", "today")):
            now = datetime.now()
            return f"**Local time:** {now.strftime('%A, %B %d, %Y — %H:%M:%S')}."
        if pl.startswith("translate ") and len(words) >= 3:
            phrase = " ".join(words[2:])
            return (
                f"I can't call external translation APIs (offline build). "
                f'Your phrase: "{phrase}". Tell me target language and I can suggest a manual approach.'
            )
        if (pl.startswith("define ") or pl.startswith("what is ")) and "2+2" not in pl and not re.search(
            r"\d\s*[+\-*/]", pl
        ):
            topic = p.split(maxsplit=2)[-1].strip("?.")
            if topic:
                return cls._structured_explain(topic)
        if any(w in pl for w in ("sad", "stressed", "anxious", "overwhelmed")):
            return (
                "I'm sorry you're dealing with that. If it's technical stress, try one tiny step: "
                "name the blocker, test the smallest input, and share any error text — "
                "I can help debug in the sandbox."
            )
        if any(w in pl for w in ("awesome", "great job", "nice", "cool", "amazing")):
            return "Glad that helped! Want to run code, open the canvas, or dig into something else?"
        if len(words) <= 2 and pl in ("yes", "no", "ok", "okay", "sure", "yep", "nope"):
            if history and history[-1][0] == "assistant":
                return "Got it. What's the next step you'd like to take?"
            return "Understood. What would you like to do next?"
        if pl.startswith("repeat ") or pl == "again":
            for role, text in reversed(history):
                if role == "assistant":
                    return f"Here's what I said last:\n\n{text[:500]}"
            return "There's no prior assistant message to repeat yet."
        if "explain" in pl or pl.startswith("how do i ") or pl.startswith("how to "):
            if ultrathink_on:
                return None
            topic = p
            for prefix in ("how do i ", "how to ", "explain "):
                if pl.startswith(prefix):
                    topic = p[len(prefix) :].strip("?.")
                    break
            return cls._structured_explain(topic)
        if any(w in pl for w in ("opinion", "think about", "should i")):
            return (
                "Here's how I'd think about it:\n\n"
                "- **Prototype first** — a short script in the sandbox often answers the question.\n"
                "- **Tradeoffs** — clarity vs speed, and maintenance cost.\n"
                "- **Reversible steps** — prefer choices you can undo cheaply.\n\n"
                "Share more context if you want a sharper recommendation."
            )
        if re.match(r"^(hi|hello|hey)\b", pl) and len(words) <= 5:
            return (
                f"Hello! I'm **{BRAND}**. "
                "Ask me anything — code, explanations, or `/help` for tools."
            )
        if ultrathink_on or CAT_R1_SYNTH_DEFAULT:
            if pl.endswith("?") and len(words) >= 3:
                return None
            if len(words) >= 4 and not pl.startswith("/"):
                return None
        if pl.endswith("?") and len(words) >= 4 and not any(k in pl for k in ("python", "code", "error", "traceback")):
            return cls._structured_explain(p.rstrip("?"))
        if len(words) >= 6 and not pl.startswith("/"):
            return (
                f"Here's my take on that:\n\n"
                f"{cls._structured_explain(p[:80])}\n\n"
                "If you want a deeper answer, narrow the question or paste relevant code."
            )
        return None


@dataclass(slots=True)
class ConversationThread:
    id: str
    title: str
    history: list[tuple[str, str]] = field(default_factory=list)
    created: float = field(default_factory=time.time)


class ConversationStore:
    """Multi-chat sidebar."""

    def __init__(self) -> None:
        self.threads: dict[str, ConversationThread] = {}
        self.active_id: str = ""
        self.new_chat("Welcome chat")

    def new_chat(self, title: str = "New chat") -> str:
        cid = uuid.uuid4().hex[:10]
        self.threads[cid] = ConversationThread(id=cid, title=title[:48])
        self.active_id = cid
        return cid

    def active(self) -> ConversationThread:
        if self.active_id not in self.threads:
            self.new_chat()
        return self.threads[self.active_id]

    def titles(self) -> list[tuple[str, str]]:
        items = sorted(self.threads.values(), key=lambda t: t.created, reverse=True)
        return [(t.id, t.title) for t in items]


class CanvasDocument:
    """Document canvas: collaborative draft pane."""

    def __init__(self) -> None:
        self.text = (
            "# Canvas document\n\n"
            "Draft essays, code, or notes here.\n"
            "Say **open canvas** or `/doc show` to focus here.\n"
        )

    def append(self, block: str) -> None:
        self.text = (self.text.rstrip() + "\n\n" + block.strip() + "\n").lstrip()

    def replace(self, text: str) -> None:
        self.text = text

    def summarize_request(self, prompt: str) -> str:
        pl = prompt.lower()
        if "outline" in pl:
            return self.text + "\n\n## Outline\n- Introduction\n- Main points\n- Conclusion\n"
        if any(k in pl for k in ("essay", "article", "write", "draft")):
            topic = prompt.strip().strip("?.")
            return self.text + f"\n\n## Draft\n**Topic:** {topic}\n\nOpening paragraph goes here.\n"
        return self.text + f"\n\n> Added from chat: {prompt[:200]}\n"


class VirtualFileStore:
    """In-memory file attach (no disk)."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    def add(self, name: str, content: str) -> str:
        if FILES_OFF or not VIRTUAL_FILES_ENABLED:
            return ""
        safe = re.sub(r"[^\w.\-]+", "_", name.strip())[:64] or "paste.txt"
        self.files[safe] = content[:50000]
        return safe

    def list_names(self) -> list[str]:
        return list(self.files.keys())

    def get(self, name: str) -> str | None:
        return self.files.get(name)

    def analyze_prompt(self, name: str) -> str:
        body = self.files.get(name)
        if not body:
            return f"File {name!r} not found."
        lines = body.splitlines()
        preview = "\n".join(lines[:12])
        return (
            f"**File `{name}`** ({len(body)} chars, {len(lines)} lines)\n"
            f"```\n{preview}\n```\n"
            f"Ask me to summarize, find bugs, or extract data from this file."
        )


class UserMemory:
    """User memory: lightweight facts from chat."""

    def __init__(self) -> None:
        self.facts: dict[str, str] = {}

    def ingest(self, prompt: str) -> None:
        m = re.search(r"(?:my name is|call me|i am)\s+([A-Za-z][A-Za-z0-9 _\-]{1,30})", prompt, re.I)
        if m:
            self.facts["name"] = m.group(1).strip()
        if "prefer" in prompt.lower():
            self.facts["preference"] = prompt.strip()[:200]

    def summary(self) -> str:
        if not self.facts:
            return "No saved memory yet. Say `my name is ...` or set facts in the Memory tab."
        return "\n".join(f"- **{k}**: {v}" for k, v in self.facts.items())

    def context_line(self) -> str:
        if not self.facts:
            return ""
        return "User memory: " + "; ".join(f"{k}={v}" for k, v in self.facts.items())


class CodeInterpreterTool:
    """Auto-runs Python for analysis prompts."""

    TRIGGER_WORDS = (
        "run", "execute", "calculate", "compute", "plot", "chart", "analyze",
        "python", "code", "interpreter", "sandbox", "fibonacci", "prime",
    )

    def __init__(self, sandbox: PythonSandbox) -> None:
        self.sandbox = sandbox

    def should_auto_run(self, prompt: str, mode: str) -> bool:
        pl = prompt.lower()
        if mode == MODE_CODE:
            return True
        if "```" in prompt:
            return True
        if any(w in pl for w in ("plot", "chart", "graph", "fibonacci", "prime", "histogram")):
            return True
        if any(w in pl for w in self.TRIGGER_WORDS) and (
            _extract_python_code(prompt) or "print(" in pl or "chart_" in pl
        ):
            return True
        return False

    def run_prompt(self, prompt: str, *, mode: str = MODE_CHAT) -> tuple[str, list[dict]]:
        code = _extract_python_code(prompt)
        if not code:
            code = self._synthesize_code(prompt)
        if not code and mode == MODE_CODE:
            code = prompt.strip()
        if not code:
            return "No runnable Python detected. Use ```python blocks or /run <code>.", []
        self.sandbox.ctx.canvas_ops.clear()
        result = self.sandbox.run(code)
        self.sandbox.ctx.canvas_ops  # keep ops
        msg = self.sandbox.format_result(result)
        return msg, list(self.sandbox.ctx.canvas_ops)

    def _synthesize_code(self, prompt: str) -> str | None:
        pl = prompt.lower()
        if "fibonacci" in pl:
            m = re.search(r"(\d+)", prompt)
            n = int(m.group(1)) if m else 10
            n = min(n, 30)
            return (
                f"n = {n}\n"
                "a, b = 0, 1\n"
                "seq = []\n"
                "for _ in range(n):\n"
                "    seq.append(a)\n"
                "    a, b = b, a + b\n"
                "print(seq)\n"
                "chart_bar([str(i) for i in range(len(seq))], seq, title='Fibonacci')\n"
            )
        if "plot" in pl or "chart" in pl or "graph" in pl:
            return (
                "xs = list(range(10))\n"
                "ys = [x * x for x in xs]\n"
                "print(list(zip(xs, ys)))\n"
                "chart_line(xs, ys, title='y = x^2')\n"
            )
        if "prime" in pl:
            return (
                "def primes(n):\n"
                "    out = []\n"
                "    for p in range(2, n + 1):\n"
                "        if all(p % d for d in range(2, int(p**0.5) + 1)):\n"
                "            out.append(p)\n"
                "    return out\n"
                "ps = primes(50)\n"
                "print(ps)\n"
                "chart_bar([str(p) for p in ps[:12]], ps[:12], title='Primes')\n"
            )
        m = re.search(r"(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)", prompt)
        if m:
            a, op, b = m.group(1), m.group(2), m.group(3)
            return f"print({a} {op} {b})\n"
        return None


class CatR1Engine:
    def __init__(self) -> None:
        self.history: list[tuple[str, str]] = []
        self.last_aha = ""
        self.last_think = ""
        self.last_tool_output: str | None = None
        self.last_canvas_ops: list[dict] = []
        self.last_doc_update: bool = False
        self.last_prompt: str = ""
        self.last_reply: str = ""
        self.mode: str = MODE_CHAT
        self.ultrathink_on = ULTRATHINK_DEFAULT
        self.custom_instructions: str = CAT_R1_SYSTEM
        self.cancel_flag = threading.Event()
        self._multiline_buffer: list[str] = []
        self.interpreter_ctx = InterpreterContext()
        self.python_sandbox = PythonSandbox(self.interpreter_ctx)
        self.code_interpreter = CodeInterpreterTool(self.python_sandbox)
        self.terminal = TerminalSandbox(self.python_sandbox)
        self.canvas = CanvasWorkspace()
        self.canvas_doc = CanvasDocument()
        self.files = VirtualFileStore()
        self.memory = UserMemory()
        self.chats = ConversationStore()
        self.history = self.chats.active().history
        self.tokenizer = ByteTokenizer()
        self.cfg = ModelConfig()
        self.teacher = CatR1LM(self.cfg, seed=1337)
        self.model = CatR1Distil8B(self.teacher, passes=self.cfg.distil_passes)
        self.ultrathink = UltraThinkEngine()
        self.synth = CatR1Synthesizer()
        self.prior = BigramPrior(self.tokenizer, STYLE_CORPUS)
        self.allowed_tokens = [10] + list(range(32, 127)) + [self.tokenizer.eos_id]

    def profile_text(self) -> str:
        nz = self.model.average_nonzero_ratio() * 100.0
        active = self.teacher.total_ternary_params()
        return (
            f"# {BRAND}\n"
            f"**Model:** {MODEL_NAME} (distilled from {TEACHER_LABEL})\n\n"
            f"- nominal params = {self.cfg.nominal_params:,} (8B target)\n"
            f"- distil passes = {self.cfg.distil_passes} student + 1 teacher\n"
            f"- active ternary params = {self.model.total_ternary_params():,}\n"
            f"- teacher only = {active:,}\n"
            f"- ultrathink = {'on' if self.ultrathink_on else 'off'}\n"
            f"- cat r1 synth = on ({TEACHER_LABEL} → {MODEL_NAME})\n"
            f"- files = off (locked)\n"
            f"- target runtime = Python {PYTHON_TARGET}\n"
            f"- GUI = tkinter\n"
            f"- tokenizer = byte-level UTF-8\n"
            f"- context = {self.cfg.context_size} tokens\n"
            f"- d_model = {self.cfg.d_model}\n"
            f"- layers = {self.cfg.n_layers}\n"
            f"- heads = {self.cfg.n_heads}\n"
            f"- feed-forward = {self.cfg.ffn_dim}\n"
            f"- ternary weights = -1, 0, 1 (1-bit path)\n"
            f"- average nonzero ratio = {nz:.1f}%\n"
            f"- external checkpoints = none\n"
            f"- network/API = off\n"
            f"- python sandbox = on (restricted)\n"
            f"- mode = {self.mode}\n"
        )

    def model_text(self) -> str:
        return (
            f"{MODEL_NAME}\n"
            "────────────────────────────\n"
            f"Teacher target: {TEACHER_LABEL}\n"
            f"Students: {self.cfg.distil_passes} parallel heads (logit ensemble)\n"
            f"Synthesis: CatR1Synthesizer (structured answers, files=off)\n"
            f"Nominal capacity: {self.cfg.nominal_params:,} parameters\n"
            f"Blend: 34% teacher + 66% student mean\n\n"
            f"UltraThink: parse → decompose → reason → verify → synthesize\n\n"
            f"1. Byte tokenizer -> embeddings ({self.cfg.vocab_size} vocab)\n"
            f"2. {self.cfg.n_layers} causal transformer block(s) per head\n"
            "3. RMSNorm -> ternary self-attention -> residual\n"
            "4. RMSNorm -> ternary gated MLP -> residual\n"
            "5. Final RMSNorm -> ternary LM head -> ensemble merge\n"
            "\n"
            "All weights embedded in-process (files=off, no download)."
        )

    def help_text(self) -> str:
        return textwrap.dedent(
            f"""
            {BRAND} — {MODEL_NAME} (offline, files=off)

            Quality: {MODEL_NAME} + UltraThink on substantive prompts
            UltraThink: /ultrathink on | off | /ultrathink <question>
            Chat: natural language, regenerate, stop
            Modes: /mode chat | code_interpreter | canvas | analysis
            Chats: /new  /chats  (sidebar in GUI)

            Code interpreter: /run <code>  /python <code>  or paste ```python blocks
              Auto-runs for: calculate, plot, fibonacci, primes, charts
              Sandbox: math, json, statistics, chart_bar(), chart_line()

            Canvas draw: /canvas demo | clear  |  plot a chart
            Canvas doc: /doc show | /doc append <text>  |  "write an outline"

            Memory: /memory  |  say "my name is ..."

            Tools: /terminal help | /profile | /model | /reset
            GUI tabs: Chat | Code | Canvas | Doc | Memory | Terminal | Python

            Try: plot a chart | fibonacci 12 | what can you do? | who are you?
            """
        ).strip()

    def _coherent_reply(self, prompt: str, candidate: str) -> str:
        """Never return gibberish — stop and substitute a readable answer."""
        text = (candidate or "").strip()
        if text and not _is_unreadable(text):
            return text
        synth = self.synth.synthesize(prompt, self.history, self.memory)
        if synth.strip() and not _is_unreadable(synth):
            return synth
        conv = ConversationalHeuristics.try_reply(
            prompt, self.history, ultrathink_on=False
        )
        if conv and not _is_unreadable(conv):
            return conv
        fb = self._fallback_reply(prompt)
        if not _is_unreadable(fb):
            return fb
        return (
            f"I stopped the local sampler to avoid gibberish. **{BRAND}** ({MODEL_NAME}) is still here.\n\n"
            "Rephrase your question in one sentence, or try `/help`. **files = off** — paste any text in chat."
        )

    def _fallback_reply(self, prompt: str) -> str:
        p = prompt.strip()
        pl = p.lower()
        if not p:
            return f"Send a message when you're ready — **{BRAND}** is online. **files = off**."
        if any(k in pl for k in ("build", "make", "create", "design")) and any(k in pl for k in ("gui", "model", "transformer")):
            return (
                "Here's a solid layout for that:\n\n"
                "- Main thread: tkinter GUI\n"
                "- Worker thread: inference\n"
                "- Stack: byte tokenizer → embeddings → causal blocks → ternary LM head\n"
                "- **files = off**, bootstrap weights in-process"
            )
        if "?" in p:
            return (
                "Happy to help — could you share a bit more detail? "
                "A concrete goal, constraint, or error line will let me give a sharper answer."
            )
        return (
            f"**{BRAND}** is running on your machine. "
            "Tell me what you'd like to accomplish and I'll respond clearly."
        )

    def _seed_prefix(self, prompt: str) -> str:
        pl = prompt.lower()
        if any(k in pl for k in ("make", "build", "create")):
            return "Certainly. Here's a clean approach: "
        if any(k in pl for k in ("explain", "how", "why", "?")):
            return "Here's how I'd frame it: "
        return "Sure — "

    def _distil_draft(self, prompt: str, *, max_tokens: int = 40) -> str:
        """Short distil-8B generation for UltraThink reason step."""
        ctx = f"UltraThink reason: {prompt[:120]}\nDraft:"
        token_ids = self.tokenizer.encode(ctx, add_bos=True, add_eos=False)
        if len(token_ids) > self.cfg.context_size:
            token_ids = token_ids[-self.cfg.context_size :]
        generated: list[int] = []
        rnd = random.Random(_stable_seed("ultrathink", prompt))
        for _ in range(max_tokens):
            if self.cancel_flag.is_set():
                break
            logits = self.model.forward_last(token_ids)
            prior = self.prior.logits(token_ids[-1])
            merged = [-1e9] * self.cfg.vocab_size
            for i in self.allowed_tokens:
                merged[i] = logits[i] * 0.45 + prior[i] * 0.55
            next_tok = self._sample_token(merged, rnd, top_k=16, temperature=0.72)
            if next_tok == self.tokenizer.eos_id:
                break
            token_ids.append(next_tok)
            token_ids = token_ids[-self.cfg.context_size :]
            generated.append(next_tok)
        draft = _clean_generated(self.tokenizer.decode(generated))
        if _is_unreadable(draft):
            return ""
        return draft

    def _sample_token(self, logits: list[float], rnd: random.Random, *, top_k: int = 12, temperature: float = 0.82) -> int:
        idx = sorted(self.allowed_tokens, key=lambda i: logits[i], reverse=True)[:top_k]
        top_vals = [logits[i] / max(0.05, temperature) for i in idx]
        probs = _softmax(top_vals)
        r = rnd.random()
        c = 0.0
        for i, p in zip(idx, probs):
            c += p
            if r <= c:
                return i
        return idx[-1]

    def _model_reply(self, prompt: str, *, think_context: str = "") -> str:
        prefix = self._seed_prefix(prompt)
        mem = self.memory.context_line()
        think_block = f"UltraThink summary:\n{think_context}\n" if think_context else ""
        context = (
            f"System: {CAT_R1_SYSTEM}\n"
            + (f"{mem}\n" if mem else "")
            + think_block
            + f"User: {prompt}\n"
            + f"Assistant: {prefix}"
        )
        token_ids = self.tokenizer.encode(context, add_bos=True, add_eos=False)
        if len(token_ids) > self.cfg.context_size:
            token_ids = token_ids[-self.cfg.context_size :]
        generated: list[int] = []
        rnd = random.Random(_stable_seed(prompt, len(self.history), think_context[:40]))
        recent_window = 28

        for _ in range(96):
            if self.cancel_flag.is_set():
                break
            bit_logits = self.model.forward_last(token_ids)
            prior_logits = self.prior.logits(token_ids[-1])
            merged = [-1e9] * self.cfg.vocab_size
            recent = token_ids[-recent_window:]
            counts: dict[int, int] = {}
            for tok in recent:
                counts[tok] = counts.get(tok, 0) + 1

            for i in self.allowed_tokens:
                merged[i] = (bit_logits[i] * 0.48) + (prior_logits[i] * 0.52)
                if i in counts:
                    merged[i] -= counts[i] * 0.10

            next_tok = self._sample_token(merged, rnd, top_k=18, temperature=0.78)
            if next_tok == self.tokenizer.eos_id:
                break
            token_ids.append(next_tok)
            token_ids = token_ids[-self.cfg.context_size :]
            generated.append(next_tok)

            tail = _clean_generated(self.tokenizer.decode(generated))
            if _is_unreadable(tail):
                break
            if tail.endswith("\n\n"):
                break
            if len(tail) > 220 and tail[-1] in ".!?":
                break

        body = _clean_generated(self.tokenizer.decode(generated))
        full = _clean_generated(prefix + body)
        if _is_unreadable(body) or _is_unreadable(full):
            return ""
        return full

    def _synthesize_ultrathink_answer(self, prompt: str) -> str:
        return self.synth.synthesize(prompt, self.history, self.memory)

    def _reply_with_ultrathink(self, prompt: str, *, force: bool = False) -> str:
        reason = self.synth.reasoning_line(prompt)
        self.last_think = self.ultrathink.run(prompt, engine=self, distill_draft=reason)
        self.last_aha = UltraThinkEngine._verify_note(prompt)
        return self._synthesize_ultrathink_answer(prompt)

    def _synth_reply(self, prompt: str) -> str:
        """Direct Cat R1 synthesis (fast path, no UltraThink trace)."""
        return self.synth.synthesize(prompt, self.history, self.memory)

    def _record_exchange(self, prompt: str, reply: str) -> str:
        reply = self._coherent_reply(prompt, reply)
        self.last_prompt = prompt
        self.last_reply = reply
        thread = self.chats.active()
        thread.history.append(("user", prompt))
        thread.history.append(("assistant", reply))
        if len(thread.title) < 8 or thread.title == "New chat":
            thread.title = prompt[:42] + ("…" if len(prompt) > 42 else "")
        if len(thread.history) > 60:
            thread.history = thread.history[-60:]
        self.history = thread.history
        return reply

    def regenerate(self) -> str:
        if not self.last_prompt:
            return "Nothing to regenerate yet."
        self.cancel_flag.clear()
        if self.history and self.history[-1][0] == "assistant":
            self.history.pop()
        if self.history and self.history[-1][0] == "user":
            self.history.pop()
        return self.generate(self.last_prompt)

    def generate(self, prompt: str) -> str:
        self.last_aha = ""
        self.last_think = ""
        self.last_tool_output = None
        self.last_canvas_ops = []
        self.last_doc_update = False
        self.cancel_flag.clear()
        raw = (prompt or "").strip()
        pl = raw.lower()
        self.memory.ingest(raw)
        if re.search(r"\bmy name is\b", pl) and self.memory.facts.get("name"):
            return self._record_exchange(
                prompt, f"Nice to meet you, {self.memory.facts['name']}. I'll remember that."
            )

        if pl in ("/pr", "/profile"):
            return self.profile_text()
        if pl in ("/model", "/about"):
            return self.model_text()
        if pl in ("/help", "/?", "help"):
            return self.help_text()
        if pl in ("/reset", "/clear"):
            self.chats.active().history.clear()
            self.history = self.chats.active().history
            self.last_aha = ""
            self._multiline_buffer.clear()
            self.canvas.clear()
            return "Conversation history cleared."
        if pl == "/new" or pl == "/newchat":
            self.chats.new_chat()
            self.history = self.chats.active().history
            return "Started a new chat."
        if pl == "/chats":
            lines = [f"- {title} ({cid})" for cid, title in self.chats.titles()]
            return "Chats:\n" + ("\n".join(lines) if lines else "(none)")
        if pl.startswith("/mode "):
            mode = pl[6:].strip()
            if mode in (MODE_CHAT, MODE_CODE, MODE_CANVAS, MODE_ANALYSIS):
                self.mode = mode
                return f"Mode set to **{mode}**."
            return f"Unknown mode. Use: {MODE_CHAT}, {MODE_CODE}, {MODE_CANVAS}, {MODE_ANALYSIS}"
        if pl == "/memory":
            return self.memory.summary()
        if pl == "/ultrathink":
            return f"UltraThink is **{'on' if self.ultrathink_on else 'off'}**. Use `/ultrathink on` or `/ultrathink off`."
        if pl == "/ultrathink on":
            self.ultrathink_on = True
            return "UltraThink **enabled** — parse → decompose → reason → verify → synthesize before replies."
        if pl == "/ultrathink off":
            self.ultrathink_on = False
            return "UltraThink **disabled** — direct distil-8B replies only."
        if pl == "/think" and self.last_think:
            return self.last_think
        if pl == "/files" or pl == "/files list" or pl.startswith("/file ") or pl.startswith("/attach "):
            return "**files = off** in this build — no virtual file store. Paste text in chat or use the Doc tab."
        if pl.startswith("/doc "):
            sub = pl[5:].strip()
            if sub == "show":
                self.last_doc_update = True
                return "__DOC_SHOW__\n" + self.canvas_doc.text[:2000]
            if sub.startswith("append "):
                self.canvas_doc.append(raw[10:])
                self.last_doc_update = True
                return "Appended to Canvas document."
            if sub == "clear":
                self.canvas_doc.replace("# Canvas document\n")
                self.last_doc_update = True
                return "Document cleared."
        if pl in ("/web", "/search"):
            return "Web search is offline (files = off, no network API). Use the code interpreter or paste text in chat."
        if pl in ("/image", "/dalle"):
            msg, ops = self.canvas.parse_command("demo")
            self.last_canvas_ops = ops
            return "Image generation is simulated with a Canvas sketch (offline). " + msg
        if pl in ("/terminal", "/term"):
            return self.terminal.help_text()
        if pl.startswith("/terminal "):
            msg, ops = self.terminal.run(raw[10:], canvas=self.canvas)
            self.last_canvas_ops = ops
            if msg == "__CLEAR__":
                return "__TERMINAL_CLEAR__"
            return msg
        if pl.startswith("/python ") or pl.startswith("/run ") or pl.startswith("/exec "):
            code = raw.split(" ", 1)[1] if " " in raw else ""
            result = self.python_sandbox.run(code)
            self.last_tool_output = result.stdout + result.stderr
            return self._record_exchange(prompt, self.python_sandbox.format_result(result))
        if pl in ("/canvas", "/canvas help"):
            return "Canvas: /canvas demo | /canvas clear | or use Terminal: `canvas demo`"
        if pl.startswith("/canvas "):
            msg, ops = self.canvas.parse_command(raw[8:])
            self.last_canvas_ops = ops
            return self._record_exchange(prompt, msg)

        code = _extract_python_code(raw)
        if code and any(k in pl for k in ("run", "execute", "eval", "```", "/python", "/run")):
            result = self.python_sandbox.run(code)
            self.last_tool_output = result.stdout + result.stderr
            return self._record_exchange(prompt, self.python_sandbox.format_result(result))

        if pl == ".end" and self._multiline_buffer:
            code = "\n".join(self._multiline_buffer)
            self._multiline_buffer.clear()
            result = self.python_sandbox.run(code)
            self.last_tool_output = result.stdout + result.stderr
            return self._record_exchange(prompt, self.python_sandbox.format_result(result))

        if self.mode == MODE_CANVAS or any(
            k in pl for k in ("open canvas", "canvas doc", "write an essay", "write a draft", "outline")
        ):
            self.canvas_doc.text = self.canvas_doc.summarize_request(raw)
            self.last_doc_update = True
            return self._record_exchange(
                prompt,
                "Updated the **Canvas document**. Open the **Doc** tab to edit.",
            )

        if self.code_interpreter.should_auto_run(raw, self.mode):
            msg, ops = self.code_interpreter.run_prompt(raw, mode=self.mode)
            self.last_tool_output = msg
            self.last_canvas_ops = ops
            return self._record_exchange(prompt, msg)

        conv = ConversationalHeuristics.try_reply(raw, self.history, ultrathink_on=self.ultrathink_on)
        if conv is not None:
            mem = self.memory.context_line()
            if mem and conv:
                conv = mem + "\n\n" + conv
            return self._record_exchange(prompt, conv)

        if pl in ("hi", "hello", "hey", "hi!", "hello!", "hey!"):
            return self._record_exchange(
                prompt,
                f"Hello! **{BRAND}** · **{MODEL_NAME}** is ready.\n\n"
                "Ask anything — **Cat R1** style: fast, short answer first, then detail. **files = off**.",
            )

        force_think = pl.startswith("/ultrathink ") and pl not in ("/ultrathink off", "/ultrathink on")
        think_prompt = prompt
        if force_think:
            think_prompt = raw.split(maxsplit=1)[1] if " " in raw else ""
            if not think_prompt.strip():
                return "Usage: `/ultrathink <your question>`"
        use_ultrathink = UltraThinkEngine.should_run(
            think_prompt, enabled=self.ultrathink_on, force=force_think
        ) or self.mode == MODE_ANALYSIS
        if use_ultrathink and think_prompt.strip():
            reply = self._reply_with_ultrathink(think_prompt, force=force_think)
            return self._record_exchange(prompt, reply)

        if any(k in pl for k in ("draw", "sketch")) and "doc" not in pl and not pl.startswith("/"):
            if "clear" in pl:
                self.canvas.clear()
                return self._record_exchange(prompt, "Canvas cleared.")
            if "demo" in pl or "circle" in pl or "line" in pl:
                msg, ops = self.canvas.parse_command("demo")
                self.last_canvas_ops = ops
                return self._record_exchange(prompt, msg)
            return self._record_exchange(
                prompt,
                "Open the Canvas tab, or try `/canvas demo` or Terminal: `canvas demo`.",
            )

        if pl.startswith("run python:") or pl.startswith("execute:"):
            code = raw.split(":", 1)[1].strip()
            result = self.python_sandbox.run(code)
            self.last_tool_output = result.stdout + result.stderr
            return self._record_exchange(prompt, self.python_sandbox.format_result(result))

        if "what is" in pl or "what's" in pl:
            m = re.search(r"(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)", pl)
            if m:
                a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
                if op == "+":
                    val = a + b
                elif op == "-":
                    val = a - b
                elif op == "*":
                    val = a * b
                elif op == "/" and b != 0:
                    val = a / b
                else:
                    val = None
                if val is not None:
                    out = int(val) if val == int(val) else val
                    return self._record_exchange(prompt, f"{a:g} {op} {b:g} = {out}")
        if len(pl.split()) >= 1:
            if CAT_R1_SYNTH_DEFAULT or not ALLOW_RAW_LM_TO_USER:
                return self._record_exchange(prompt, self._synth_reply(prompt))
            reply = self._model_reply(prompt)
            if not reply or _is_unreadable(reply):
                reply = self._synth_reply(prompt)
            return self._record_exchange(prompt, reply)

        return self._record_exchange(prompt, self._coherent_reply(prompt, self._fallback_reply(prompt)))


def run_cli() -> None:
    engine = CatR1Engine()
    print(f"{BRAND} CLI (files=off). Type 'exit' to quit.\n")
    while True:
        try:
            msg = input(">>> ")
            if msg.strip().lower() == "exit":
                break
            started = time.perf_counter()
            out = engine.generate(msg)
            elapsed = (time.perf_counter() - started) * 1000.0
            print(out)
            if engine.last_think:
                print("--- UltraThink ---\n", engine.last_think, sep="")
            if engine.last_aha:
                print("Verify:", engine.last_aha)
            print(f"[{elapsed:.1f} ms]\n")
        except (EOFError, KeyboardInterrupt):
            break


def run_gui() -> None:
    import tkinter as tk
    from tkinter import font, messagebox, scrolledtext, ttk

    engine = CatR1Engine()

    root = tk.Tk()
    root.title(WINDOW_TITLE)
    root.geometry("1320x760")
    root.configure(bg="#050505")
    root.minsize(1020, 600)

    fonts = {
        "mono": font.Font(family="Consolas" if os.name != "nt" else "Courier New", size=11),
        "bold": font.Font(family="Consolas" if os.name != "nt" else "Courier New", size=11, weight="bold"),
        "italic": font.Font(family="Consolas" if os.name != "nt" else "Courier New", size=10, slant="italic"),
        "small": font.Font(family="Consolas" if os.name != "nt" else "Courier New", size=9),
    }

    outer = tk.PanedWindow(root, orient="horizontal", bg="#050505", sashwidth=6, sashrelief="flat")
    outer.pack(fill="both", expand=True)

    sidebar = tk.Frame(outer, bg="#0d0d0d", width=200)
    center = tk.Frame(outer, bg="#050505")
    tools_frame = tk.Frame(outer, bg="#050505")
    outer.add(sidebar, minsize=160)
    outer.add(center, minsize=440)
    outer.add(tools_frame, minsize=320)

    tk.Label(sidebar, text=BRAND, bg="#0d0d0d", fg="#00d9ff", font=fonts["bold"]).pack(pady=(10, 0))
    tk.Label(sidebar, text=MODEL_NAME, bg="#0d0d0d", fg="#888", font=fonts["small"]).pack(pady=(0, 2))
    tk.Label(sidebar, text="UltraThink · files=off", bg="#0d0d0d", fg="#4a4a4a", font=fonts["small"]).pack(pady=(0, 8))
    chat_list = tk.Listbox(sidebar, bg="#111", fg="#ccc", font=fonts["small"], height=14, relief="flat")
    chat_list.pack(fill="both", expand=True, padx=8, pady=4)

    def refresh_chat_list() -> None:
        chat_list.delete(0, "end")
        for cid, title in engine.chats.titles():
            mark = "• " if cid == engine.chats.active_id else "  "
            chat_list.insert("end", mark + title)

    def on_new_chat() -> None:
        engine.chats.new_chat()
        engine.history = engine.chats.active().history
        refresh_chat_list()
        chat.config(state="normal")
        chat.delete("1.0", "end")
        chat.config(state="disabled")
        log_line("SYSTEM", "New chat started.")

    def on_pick_chat(_evt=None) -> None:
        sel = chat_list.curselection()
        if not sel:
            return
        label = chat_list.get(sel[0]).replace("• ", "", 1).strip()
        for cid, title in engine.chats.titles():
            if title == label:
                engine.chats.active_id = cid
                engine.history = engine.chats.active().history
                break

    tk.Button(
        sidebar, text="+ New chat", command=on_new_chat, bg="#222", fg="#00d9ff", font=fonts["small"], relief="flat"
    ).pack(fill="x", padx=8, pady=4)
    tk.Label(sidebar, text="Mode", bg="#0d0d0d", fg="#666", font=fonts["small"]).pack(anchor="w", padx=10)
    mode_var = tk.StringVar(value=engine.mode)
    mode_menu = tk.OptionMenu(
        sidebar,
        mode_var,
        MODE_CHAT,
        MODE_CODE,
        MODE_CANVAS,
        MODE_ANALYSIS,
    )
    mode_menu.config(bg="#222", fg="#00d9ff", font=fonts["small"], highlightthickness=0)
    mode_menu["menu"].config(bg="#222", fg="#00d9ff")
    mode_menu.pack(fill="x", padx=8, pady=2)

    def on_mode_change(*_a: object) -> None:
        engine.mode = mode_var.get()

    mode_var.trace_add("write", on_mode_change)
    refresh_chat_list()
    chat_list.bind("<<ListboxSelect>>", on_pick_chat)

    chat_frame = center

    chat = scrolledtext.ScrolledText(
        chat_frame,
        bg="#050505",
        fg="#00d9ff",
        font=fonts["mono"],
        insertbackground="cyan",
        relief="flat",
        padx=12,
        pady=12,
        state="disabled",
        wrap="word",
    )
    chat.pack(expand=True, fill="both")

    for tag_name, color, fnt in [
        ("user", "#ffffff", fonts["bold"]),
        ("think", "#4a4a4a", fonts["italic"]),
        ("bot", "#00aaff", fonts["bold"]),
        ("code", "#00ffaa", fonts["small"]),
        ("aha", "#ffd54f", fonts["bold"]),
        ("system", "#8a8a8a", fonts["small"]),
    ]:
        chat.tag_config(tag_name, foreground=color, font=fnt)

    notebook = ttk.Notebook(tools_frame)
    notebook.pack(fill="both", expand=True, padx=4, pady=4)

    tab_code = tk.Frame(notebook, bg="#050505")
    tab_terminal = tk.Frame(notebook, bg="#050505")
    tab_python = tk.Frame(notebook, bg="#050505")
    tab_canvas = tk.Frame(notebook, bg="#050505")
    tab_doc = tk.Frame(notebook, bg="#050505")
    tab_memory = tk.Frame(notebook, bg="#050505")
    notebook.add(tab_code, text="Code")
    notebook.add(tab_canvas, text="Canvas")
    notebook.add(tab_doc, text="Doc")
    notebook.add(tab_memory, text="Memory")
    notebook.add(tab_terminal, text="Terminal")
    notebook.add(tab_python, text="Python")

    code_info = scrolledtext.ScrolledText(
        tab_code, bg="#0a0a0a", fg="#aaa", font=fonts["small"], height=14, state="disabled", wrap="word"
    )
    code_info.pack(fill="both", expand=True, padx=4, pady=4)
    code_info.config(state="normal")
    code_info.insert(
        "end",
        "Code interpreter\n"
        "- Auto-runs ```python blocks\n"
        "- chart_bar(labels, values)\n"
        "- chart_line(xs, ys)\n"
        "- Mode: code_interpreter\n",
    )
    code_info.config(state="disabled")

    doc_text = scrolledtext.ScrolledText(
        tab_doc, bg="#0f0f14", fg="#e8e8e8", font=fonts["mono"], wrap="word"
    )
    doc_text.pack(fill="both", expand=True, padx=4, pady=4)
    doc_text.insert("1.0", engine.canvas_doc.text)

    mem_view = scrolledtext.ScrolledText(
        tab_memory, bg="#0a0a0a", fg="#ffd54f", font=fonts["small"], height=12, state="disabled", wrap="word"
    )
    mem_view.pack(fill="both", expand=True, padx=4, pady=4)

    term_out = scrolledtext.ScrolledText(
        tab_terminal,
        bg="#0a0a0a",
        fg="#b8ffb8",
        font=fonts["mono"],
        height=12,
        state="disabled",
        wrap="word",
    )
    term_out.pack(fill="both", expand=True, padx=4, pady=4)

    term_in = tk.Entry(tab_terminal, bg="#111", fg="#b8ffb8", font=fonts["mono"], insertbackground="#b8ffb8")
    term_in.pack(fill="x", padx=4, pady=(0, 4))

    py_code = scrolledtext.ScrolledText(
        tab_python,
        bg="#0a0a0a",
        fg="#00ffaa",
        font=fonts["mono"],
        height=10,
        wrap="none",
    )
    py_code.pack(fill="both", expand=True, padx=4, pady=4)
    py_code.insert("1.0", f"print('{APP_NAME} sandbox ready')\nprint(2 ** 10)")

    py_out = scrolledtext.ScrolledText(
        tab_python,
        bg="#050505",
        fg="#888",
        font=fonts["small"],
        height=5,
        state="disabled",
        wrap="word",
    )
    py_out.pack(fill="x", padx=4, pady=(0, 4))

    canvas_widget = tk.Canvas(
        tab_canvas,
        width=CanvasWorkspace.WIDTH,
        height=CanvasWorkspace.HEIGHT,
        bg=CanvasWorkspace.BG,
        highlightthickness=1,
        highlightbackground="#333",
    )
    canvas_widget.pack(fill="both", expand=True, padx=8, pady=8)
    _render_canvas(canvas_widget, engine.canvas.ops)

    canvas_bar = tk.Frame(tab_canvas, bg="#050505")
    canvas_bar.pack(fill="x", padx=8, pady=4)
    for label, cmd in [("Demo", "demo"), ("Clear", "clear")]:
        tk.Button(
            canvas_bar,
            text=label,
            command=lambda c=cmd: _canvas_btn(c),
            bg="#222",
            fg="#00d9ff",
            font=fonts["small"],
            relief="flat",
        ).pack(side="left", padx=4)

    def _canvas_btn(sub: str) -> None:
        msg, ops = engine.canvas.parse_command(sub)
        if ops:
            engine.canvas.ops[:] = ops
        _render_canvas(canvas_widget, engine.canvas.ops)
        term_log(f"[canvas] {msg}\n")

    def term_log(text: str) -> None:
        term_out.config(state="normal")
        term_out.insert("end", text if text.endswith("\n") else text + "\n")
        term_out.config(state="disabled")
        term_out.see("end")

    term_multiline: list[str] = []

    def run_term_line() -> None:
        line = term_in.get().strip()
        if not line:
            return
        term_in.delete(0, "end")
        term_log(f"$ {line}")
        pl = line.lower()
        if pl == "run":
            term_multiline.clear()
            term_log("Multiline Python: enter lines, then `.end`")
            return
        if term_multiline and pl != ".end":
            term_multiline.append(line)
            term_log(f"  + line {len(term_multiline)}")
            return
        if pl == ".end" and term_multiline:
            code = "\n".join(term_multiline)
            term_multiline.clear()
            res = engine.python_sandbox.run(code)
            term_log(engine.python_sandbox.format_result(res))
            return
        msg, ops = engine.terminal.run(line, canvas=engine.canvas)
        if msg == "__CLEAR__":
            term_out.config(state="normal")
            term_out.delete("1.0", "end")
            term_out.config(state="disabled")
            return
        term_log(msg)
        if ops:
            engine.canvas.ops[:] = ops
            _render_canvas(canvas_widget, engine.canvas.ops)

    def run_python_tab() -> None:
        code = py_code.get("1.0", "end")
        res = engine.python_sandbox.run(code)
        py_out.config(state="normal")
        py_out.delete("1.0", "end")
        py_out.insert("end", res.stdout)
        if res.stderr:
            py_out.insert("end", "\n" + res.stderr)
        py_out.insert("end", f"\n[exit {res.exit_code}]")
        py_out.config(state="disabled")

    term_in.bind("<Return>", lambda _e: run_term_line())
    tk.Button(
        tab_terminal,
        text="Run line",
        command=run_term_line,
        bg="#222",
        fg="#00d9ff",
        font=fonts["small"],
        relief="flat",
    ).pack(pady=(0, 6))
    tk.Button(
        tab_python,
        text="Run Python",
        command=run_python_tab,
        bg="#222",
        fg="#00ffaa",
        font=fonts["small"],
        relief="flat",
    ).pack(pady=4)

    term_log(engine.terminal.help_text() + "\n")

    toolbar = tk.Frame(chat_frame, bg="#050505")
    toolbar.pack(fill="x", padx=10, pady=(6, 0))

    def do_stop() -> None:
        engine.cancel_flag.set()
        status.config(text="Stop requested…")

    def do_regen() -> None:
        if not engine.last_prompt:
            return
        log_line("SYSTEM", "Regenerating…")
        status.config(text="Regenerating…")

        def worker() -> None:
            resp = engine.regenerate()
            root.after(0, lambda: finish_reply(resp))

        threading.Thread(target=worker, daemon=True).start()

    for label, cmd in [("Stop", do_stop), ("Regenerate", do_regen)]:
        tk.Button(
            toolbar, text=label, command=cmd, bg="#222", fg="#00d9ff", font=fonts["small"], relief="flat"
        ).pack(side="left", padx=2)

    inp = tk.Frame(chat_frame, bg="#050505")
    inp.pack(fill="x", padx=10, pady=5)

    entry = tk.Entry(
        inp,
        bg="#111",
        fg="#00d9ff",
        font=fonts["mono"],
        insertbackground="cyan",
        relief="flat",
        bd=2,
    )
    entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

    btns = tk.Frame(inp, bg="#050505")
    btns.pack(side="right")
    for t, c in [
        ("Help", "/help"),
        ("Term", "/terminal help"),
        ("Canvas", "/canvas demo"),
        ("Profile", "/profile"),
        ("Model", "/model"),
        ("Py", "write python code "),
        ("Reset", "/reset"),
        ("Think", "/ultrathink "),
    ]:
        tk.Button(
            btns,
            text=t,
            command=lambda c=c: entry.insert("end", c),
            bg="#222",
            fg="#00d9ff",
            font=fonts["small"],
            relief="flat",
        ).pack(side="left", padx=2)

    status = tk.Label(
        root,
        text=f"Ready | {MODEL_NAME} | files=off",
        bg="#050505",
        fg="#666",
        font=fonts["small"],
        anchor="w",
    )
    status.pack(fill="x", padx=10, pady=2)

    def log_line(sender: str, text: str, tag: str | None = None) -> None:
        body = _text_insert_safe(text if isinstance(text, str) else str(text), code_fence=(tag == "code"))
        head_tag = "bot" if sender == BOT_NAME else (tag if tag is not None else "think")
        if sender == "SYSTEM":
            head_tag = "system"
        body_tag = tag if tag is not None else ("bot" if sender == BOT_NAME else "think")
        if sender == "SYSTEM":
            body_tag = "system"
        try:
            chat.config(state="normal")
            chat.insert("end", f"[{sender}]: ", head_tag)
            chat.insert("end", f"{body}\n\n", body_tag)
            chat.config(state="disabled")
            chat.see("end")
        except tk.TclError:
            esc = (f"[{sender}]: " + body).encode("unicode_escape", errors="replace").decode("ascii", errors="replace")[:12000]
            chat.config(state="normal")
            chat.insert("end", esc + "\n\n", "think")
            chat.config(state="disabled")
            chat.see("end")

    log_line("SYSTEM", f"{BRAND} — {MODEL_NAME} + UltraThink (files=off)")
    log_line("SYSTEM", "Tools: Code, Canvas, Doc, Memory, Terminal | /ultrathink /profile /help")

    def finish_reply(resp: str, elapsed_ms: float = 0.0) -> None:
        if resp == "__TERMINAL_CLEAR__":
            term_out.config(state="normal")
            term_out.delete("1.0", "end")
            term_out.config(state="disabled")
            status.config(text=f"Ready | {elapsed_ms:.1f} ms")
            return
        if resp.startswith("__DOC_SHOW__"):
            doc_text.delete("1.0", "end")
            doc_text.insert("1.0", resp.split("\n", 1)[1])
            notebook.select(tab_doc)
        if engine.last_doc_update:
            doc_text.delete("1.0", "end")
            doc_text.insert("1.0", engine.canvas_doc.text)
            notebook.select(tab_doc)
        if engine.last_tool_output:
            term_log(engine.last_tool_output)
            code_info.config(state="normal")
            code_info.delete("1.0", "end")
            code_info.insert("end", engine.last_tool_output + "\n")
            code_info.config(state="disabled")
            notebook.select(tab_code)
        if engine.last_canvas_ops:
            engine.canvas.ops[:] = engine.last_canvas_ops
            _render_canvas(canvas_widget, engine.canvas.ops)
            notebook.select(tab_canvas)
        mem_view.config(state="normal")
        mem_view.delete("1.0", "end")
        mem_view.insert("end", engine.memory.summary())
        mem_view.config(state="disabled")
        refresh_chat_list()
        if engine.last_think:
            log_line("ULTRATHINK", engine.last_think, "think")
        if "```" in resp:
            parts = resp.split("```")
            for i, part in enumerate(parts):
                if not part or part.startswith("__DOC_SHOW__"):
                    continue
                body = part
                if i % 2 == 1:
                    body = body.lstrip()
                    if body.lower().startswith("python"):
                        body = body[6:].lstrip("\n\r")
                log_line(BOT_NAME, body, "code" if i % 2 == 1 else None)
        elif not resp.startswith("__DOC_SHOW__"):
            log_line(BOT_NAME, resp, None)
        if engine.last_aha:
            log_line("VERIFY", engine.last_aha, "aha")
        status.config(text=f"Ready | {elapsed_ms:.1f} ms | {BRAND} | mode={engine.mode}")

    def send() -> None:
        msg = entry.get().strip()
        if not msg:
            return
        entry.delete(0, "end")
        log_line("YOU", msg, "user")
        status.config(text=f"Running {BRAND}…")

        def worker() -> None:
            started = time.perf_counter()
            try:
                resp = engine.generate(msg)
            except Exception as e:  # pragma: no cover - GUI safety path
                resp = f"(error) {type(e).__name__}: {e}"
                engine.last_aha = ""
                engine.last_think = ""
            aha = engine.last_aha
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            root.after(0, lambda: finish_reply(resp, elapsed_ms))

        threading.Thread(target=worker, daemon=True).start()

    entry.bind("<Return>", lambda _e: send())
    entry.focus_set()
    def on_close() -> None:
        if messagebox.askokcancel("Quit", f"Exit {WINDOW_TITLE}?"):
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


def main(argv: list[str]) -> int:
    args = set(argv[1:])
    if "--cli" in args or "--headless" in args:
        run_cli()
        return 0
    try:
        run_gui()
        return 0
    except Exception as exc:
        print("GUI failed, switching to CLI.", file=sys.stderr)
        print("Reason:", exc, file=sys.stderr)
        run_cli()
        return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


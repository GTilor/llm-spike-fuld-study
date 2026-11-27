"""Modal A10G Benchmark Suite: Single-Pass Domain Shift (`Code -> PG-19 Literature`) & Heavy-Tailed Shocks.

Executes:
  1. Strict 3-way book/file split (`Train` / `Dev` for LR tuning / `Held-Out Test` for final evaluation),
     vocabulary fitted strictly on `Train` splits, SHA-256 corpus manifests, and all-token vs. non-UNK metrics.
  2. Step-level coordinate peak-exceedance telemetry Pr(g != 0, |z| > z_*), awakening-trap verification
     (`C11` rho_prior=0 vs. `T13` rho_prior=0.15), 1-step shock micro-responses, and GPU throughput (`tokens/s`).
  3. Equal 4-LR grid `{1.2e-3, 2.5e-3, 5.0e-3, 8.0e-3}` tuned on `Dev` books across all 10 optimizers,
     followed by evaluation across 8 fresh paired seeds `[101..808]` on `Held-Out Test` books under
     `R_clean`, `R_light_gauss`, and `R_heavy_sas` with Holm-Bonferroni family-wise error control.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import pathlib
import time
import urllib.request
from collections import Counter

import modal

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PG19_PARQUET_URL = "https://huggingface.co/api/datasets/emozilla/pg19-test/parquet/default/test/0.parquet"


def download_pg19_cached():
    os.makedirs("/cache_data", exist_ok=True)
    target = "/cache_data/pg19_test.parquet"
    if not os.path.exists(target):
        req = urllib.request.Request(PG19_PARQUET_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(target, "wb") as f:
            f.write(r.read())


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "scipy", "tiktoken", "pyarrow")
    .run_function(download_pg19_cached)
    .add_local_file(str(REPO_ROOT / "adafuld" / "__init__.py"), remote_path="/root/adafuld/__init__.py")
    .add_local_file(str(REPO_ROOT / "adafuld" / "optimizers.py"), remote_path="/root/adafuld/optimizers.py")
)

app = modal.App("llm-spike-fuld-study", image=image)

DEV_SEEDS = [11, 22]
TEST_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808]
LR_GRID = [1.2e-3, 2.5e-3, 5.0e-3, 8.0e-3]
SEQ_LEN = 128
BATCH_SIZE = 32   # 4,096 tokens/step
EVAL_BATCHES = 32  # 131,072 target tokens per split evaluation

TUNED_METHODS = [
    "C01_AdamW",
    "B16_GradClip",
    "B17_StableAdamW_Exact",
    "B18_LargeEps_1e-4",
    "N19_CoordClip",
    "N20_ExactTAdamW",
    "C05_FlatTop_raw",
    "C09_FULD_raw",
    "T13_FlatTop",
    "T13_AdaFULD_PriorFloor",
]


def _prepare_3way_corpora(device):
    """Split Python source files and PG-19 books into disjoint Train / Dev / Held-Out Test sets."""
    import numpy as np
    import pyarrow.parquet as pq
    import tiktoken
    import torch

    py_files: list[str] = []
    py_roots = [
        pathlib.Path(os.__file__).parent,
        pathlib.Path(torch.__file__).parent.parent,
    ]
    total_chars = 0
    for root in py_roots:
        for fp in sorted(root.rglob("*.py")):
            try:
                txt = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if len(txt) > 500:
                py_files.append(txt)
                total_chars += len(txt)
            if total_chars >= 26_000_000:
                break
        if total_chars >= 26_000_000:
            break

    n_cf = len(py_files)
    i_c_dev = int(n_cf * 0.76)
    i_c_tst = int(n_cf * 0.88)
    raw_code_tr = "\n\n".join(py_files[:i_c_dev])[:18_000_000]
    raw_code_dv = "\n\n".join(py_files[i_c_dev:i_c_tst])[:2_500_000]
    raw_code_te = "\n\n".join(py_files[i_c_tst:])[:2_500_000]

    tbl = pq.read_table("/cache_data/pg19_test.parquet", columns=["text"])
    books = [str(x) for x in tbl["text"].to_pylist() if len(str(x)) > 5000]
    n_bk = len(books)
    i_b_dev = int(n_bk * 0.68)
    i_b_tst = int(n_bk * 0.84)
    raw_lit_tr = "\n\n".join(books[:i_b_dev])[:20_000_000]
    raw_lit_dv = "\n\n".join(books[i_b_dev:i_b_tst])[:3_500_000]
    raw_lit_te = "\n\n".join(books[i_b_tst:])[:3_500_000]

    enc = tiktoken.get_encoding("gpt2")
    c_tr_bpe = enc.encode_ordinary(raw_code_tr)
    c_dv_bpe = enc.encode_ordinary(raw_code_dv)
    c_te_bpe = enc.encode_ordinary(raw_code_te)
    l_tr_bpe = enc.encode_ordinary(raw_lit_tr)
    l_dv_bpe = enc.encode_ordinary(raw_lit_dv)
    l_te_bpe = enc.encode_ordinary(raw_lit_te)

    # Fit 8,192 BPE vocabulary strictly on (Code Train + Lit Train)
    VOCAB_SIZE = 8192
    byte_tokens = [enc.encode_single_token(bytes([b])) for b in range(256) if bytes([b]) in enc._mergeable_ranks]
    kept_set = set(byte_tokens)
    c_ranked = [tok for tok, _ in Counter(c_tr_bpe).most_common()]
    l_ranked = [tok for tok, _ in Counter(l_tr_bpe).most_common()]
    ci, li = 0, 0
    while len(kept_set) < VOCAB_SIZE - 1 and (ci < len(c_ranked) or li < len(l_ranked)):
        if ci < len(c_ranked):
            kept_set.add(c_ranked[ci])
            ci += 1
        if len(kept_set) < VOCAB_SIZE - 1 and li < len(l_ranked):
            kept_set.add(l_ranked[li])
            li += 1

    kept_list = sorted(kept_set)
    UNK_ID = len(kept_list)
    ACTUAL_VOCAB = UNK_ID + 1
    lut = np.full(enc.n_vocab + 1, UNK_ID, dtype=np.int64)
    for new_id, old_id in enumerate(kept_list):
        lut[old_id] = new_id

    def map_t(bpe_list: list[int]) -> torch.Tensor:
        return torch.from_numpy(lut[np.asarray(bpe_list, dtype=np.int64)]).to(device)

    splits = {
        "code_train": map_t(c_tr_bpe),
        "code_dev": map_t(c_dv_bpe),
        "code_test": map_t(c_te_bpe),
        "lit_train": map_t(l_tr_bpe),
        "lit_dev": map_t(l_dv_bpe),
        "lit_test": map_t(l_te_bpe),
    }
    manifest = {
        "vocab_size": ACTUAL_VOCAB,
        "unk_id": UNK_ID,
        "pg19_books_total": n_bk,
        "pg19_books_split": {"train": i_b_dev, "dev": i_b_tst - i_b_dev, "test": n_bk - i_b_tst},
        "py_files_split": {"train": i_c_dev, "dev": i_c_tst - i_c_dev, "test": n_cf - i_c_tst},
        "tokens": {k: int(v.numel()) for k, v in splits.items()},
        "sha256_prefix16": {
            "lit_train": hashlib.sha256(raw_lit_tr[:100_000].encode()).hexdigest()[:16],
            "lit_dev": hashlib.sha256(raw_lit_dv[:100_000].encode()).hexdigest()[:16],
            "lit_test": hashlib.sha256(raw_lit_te[:100_000].encode()).hexdigest()[:16],
        },
        "lit_test_unk_pct": round(100.0 * float((splits["lit_test"] == UNK_ID).float().mean().item()), 3),
    }
    return splits, manifest


def _build_model_cls():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class CausalSelfAttention(nn.Module):
        def __init__(self, d_model: int, n_head: int):
            super().__init__()
            self.n_head = n_head
            self.head_dim = d_model // n_head
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.proj = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, T, C = x.shape
            q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).unbind(dim=2)
            y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
            return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))

    class Block(nn.Module):
        def __init__(self, d_model: int, n_head: int):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = CausalSelfAttention(d_model, n_head)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model, bias=False),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model, bias=False),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x

    class MiniGPT(nn.Module):
        def __init__(self, vocab_size: int, d_model: int = 320, n_layer: int = 6, n_head: int = 5):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(SEQ_LEN, d_model)
            self.blocks = nn.ModuleList([Block(d_model, n_head) for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            self.apply(self._init_weights)

        def _init_weights(self, m: nn.Module):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
            B, T = idx.shape
            x = self.tok_emb(idx) + self.pos_emb(torch.arange(0, T, device=idx.device))
            for blk in self.blocks:
                x = blk(x)
            logits = self.lm_head(self.ln_f(x))
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

    return MiniGPT


def _slice_batch(tokens, token_offset: int, batch_size: int = BATCH_SIZE):
    span = batch_size * (SEQ_LEN + 1)
    if token_offset + span > tokens.numel():
        raise RuntimeError(f"1-pass violation: {token_offset} + {span} > {tokens.numel()}")
    buf = tokens[token_offset : token_offset + span].view(batch_size, SEQ_LEN + 1)
    return buf[:, :-1].contiguous(), buf[:, 1:].contiguous()


def _build_paired_shock_schedules(seed: int, n_steps: int, n_tensors: int):
    """Construct timestamp-, mask-, and sign-matched Light (Gaussian) vs. Heavy (SαS α=1.2) shock schedules.

    Note: Preserves the exact constant `GAUSS_MEDIAN_MATCH_SCALE = 3.9265` used to produce
    `results/exp_results_v3.json` (empirical median |xi| ≈ 2.63 light vs. 2.98 heavy; for exact
    CMS unit-scale median 0.981537 matching, use 3.0 * 0.981537 / 0.67449 = 4.3657).
    """
    import torch

    GAUSS_MEDIAN_MATCH_SCALE = 3.9265

    def sample_sas12(gen: torch.Generator) -> float:
        u = (torch.rand(1, generator=gen).item() - 0.5) * math.pi
        w = -math.log(max(1e-12, torch.rand(1, generator=gen).item()))
        return (math.sin(1.2 * u) / (math.cos(u) ** (1.0 / 1.2))) * (
            (math.cos(-0.2 * u) / w) ** (-0.2 / 1.2)
        )

    mask_gen = torch.Generator().manual_seed(seed + 50000)
    sas_gen = torch.Generator().manual_seed(seed + 60000)
    gau_gen = torch.Generator().manual_seed(seed + 70000)

    light_sched = []
    heavy_sched = []
    for _ in range(n_steps):
        if torch.rand(1, generator=mask_gen).item() < 0.15:
            l_vec, h_vec = [], []
            for _ in range(n_tensors):
                if torch.rand(1, generator=mask_gen).item() < 0.25:
                    sgn = 1.0 if torch.rand(1, generator=mask_gen).item() < 0.5 else -1.0
                    mag_h = min(300.0, abs(sample_sas12(sas_gen))) * 3.0
                    mag_l = abs(float(torch.randn(1, generator=gau_gen).item())) * GAUSS_MEDIAN_MATCH_SCALE
                    h_vec.append(sgn * mag_h)
                    l_vec.append(sgn * mag_l)
                else:
                    h_vec.append(0.0)
                    l_vec.append(0.0)
            light_sched.append(l_vec)
            heavy_sched.append(h_vec)
        else:
            light_sched.append(None)
            heavy_sched.append(None)
    return light_sched, heavy_sched


@app.function(gpu="A10G", timeout=1500)
def run_step_telemetry() -> dict:
    """Measure step-by-step coordinate exceedance, awakening trap rates, 1-step Δθ norm, and A10G throughput."""
    import sys

    sys.path.insert(0, "/root")

    import numpy as np
    import torch
    from adafuld import eval_flattop_twin, eval_psi_unit, exact_peak_location, make_optimizer

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    splits, manifest = _prepare_3way_corpora(device)
    MiniGPT = _build_model_cls()
    vocab_size = manifest["vocab_size"]
    span32 = BATCH_SIZE * (SEQ_LEN + 1)

    torch.manual_seed(101)
    base_m = MiniGPT(vocab_size).to(device)
    base_o = make_optimizer("C01_AdamW", base_m.parameters(), lr=5e-3)
    for s in range(600):
        base_o.param_groups[0]["lr"] = 5e-3 * min(1.0, (s + 1) / 50.0)
        xb, yb = _slice_batch(splits["code_train"], s * span32)
        base_o.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = base_m(xb, yb)
        loss.backward()
        base_o.step()

    code_prefix_counts = torch.bincount(splits["code_train"][: 600 * span32].cpu(), minlength=vocab_size).to(device)
    unseen_vocab_mask = (code_prefix_counts == 0) & (torch.arange(vocab_size, device=device) != manifest["unk_id"])

    @torch.no_grad()
    def eval_fast_test(m) -> float:
        m.eval()
        tot = 0.0
        for b in range(16):
            xb, yb = _slice_batch(splits["lit_test"], b * span32)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, l = m(xb, yb)
            tot += float(l.item())
        m.train()
        return tot / 16.0

    _, heavy_sched = _build_paired_shock_schedules(101, 120, len(list(base_m.parameters())))
    z_star = exact_peak_location(1.5)

    ref_m = copy.deepcopy(base_m)
    ref_o = make_optimizer("T13_AdaFULD_PriorFloor", ref_m.parameters(), lr=3.5e-3)
    for p_b, p_t in zip(base_m.parameters(), ref_m.parameters()):
        st_b = base_o.state[p_b]
        ref_o.state[p_t] = {
            "step": int(st_b["step"]),
            "exp_avg": st_b["exp_avg"].clone(),
            "exp_avg_sq": st_b["exp_avg_sq"].clone(),
        }

    hit_exceed_t13, unhit_exceed_t13 = [], []
    hit_energy_gap, unhit_energy_gap = [], []
    unseen_exceed_no_floor, unseen_exceed_with_floor = [], []
    shock_micro_responses = []

    for s in range(80):
        xb, yb = _slice_batch(splits["lit_train"], s * span32)
        ref_o.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, l = ref_m(xb, yb)
        l.backward()

        ev = heavy_sched[s]
        if ev is not None:
            g_main = [p.grad.detach().clone() for p in ref_m.parameters()]
            xb_a, yb_a = _slice_batch(splits["lit_train"], (420 + s) * span32)
            ref_o.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, l_a = ref_m(xb_a, yb_a)
            l_a.backward()
            for p, gm, xi in zip(ref_m.parameters(), g_main, ev):
                if xi != 0.0:
                    p.grad.copy_(gm + xi * (p.grad.detach() - gm))
                else:
                    p.grad.copy_(gm)

        with torch.no_grad():
            hit_nz, hit_ex, unhit_nz, unhit_ex = 0, 0, 0, 0
            hit_eg_sum, unhit_eg_sum = 0.0, 0.0
            for idx_p, p in enumerate(ref_m.parameters()):
                g = p.grad.float()
                st = ref_o.state[p]
                t_prev = st["step"]
                m_hat = st["exp_avg"] / (1.0 - 0.9 ** t_prev)
                v_hat = st["exp_avg_sq"] / (1.0 - 0.995 ** t_prev)
                sqrt_v = torch.sqrt(v_hat)
                s_prior = torch.maximum(sqrt_v, 0.15 * torch.sqrt(torch.mean(v_hat))) + 1e-8
                h = 3.0 * s_prior
                z = (g - m_hat) / h
                nz = g != 0.0
                ex = nz & (z.abs() > z_star)
                P_val = eval_flattop_twin(z, 1.5)
                psi_val = eval_psi_unit(z, 1.5)
                eg = torch.where(nz, h.pow(2) * (P_val.pow(2) - psi_val.pow(2)), torch.zeros_like(h))

                is_hit = (ev is not None) and (ev[idx_p] != 0.0)
                if is_hit:
                    hit_nz += int(nz.sum().item())
                    hit_ex += int(ex.sum().item())
                    hit_eg_sum += float(eg.sum().item())
                else:
                    unhit_nz += int(nz.sum().item())
                    unhit_ex += int(ex.sum().item())
                    unhit_eg_sum += float(eg.sum().item())

                if p is ref_m.tok_emb.weight and s < 25:
                    g_u = g[unseen_vocab_mask]
                    nz_u = g_u != 0.0
                    if nz_u.any():
                        mh_u = m_hat[unseen_vocab_mask]
                        sv_u = sqrt_v[unseen_vocab_mask]
                        z_no_f = (g_u - mh_u) / (3.0 * (sv_u + 1e-8))
                        z_wi_f = (g_u - mh_u) / h[unseen_vocab_mask]
                        unseen_exceed_no_floor.append(100.0 * float((z_no_f[nz_u].abs() > z_star).float().mean().item()))
                        unseen_exceed_with_floor.append(100.0 * float((z_wi_f[nz_u].abs() > z_star).float().mean().item()))

            if hit_nz > 0:
                hit_exceed_t13.append(100.0 * hit_ex / hit_nz)
                hit_energy_gap.append(hit_eg_sum / hit_nz)
            if unhit_nz > 0:
                unhit_exceed_t13.append(100.0 * unhit_ex / unhit_nz)
                unhit_energy_gap.append(unhit_eg_sum / unhit_nz)

            if ev is not None and len(shock_micro_responses) < 4 and any(x != 0.0 for x in ev):
                loss_before = eval_fast_test(ref_m)
                step_resp = {"step": s + 1, "max_abs_xi": round(max(abs(x) for x in ev), 2), "arms": {}}
                for arm_probe in [
                    "C01_AdamW",
                    "B16_GradClip",
                    "B18_LargeEps_1e-4",
                    "T13_FlatTop",
                    "T13_AdaFULD_PriorFloor",
                    "C09_FULD_raw",
                ]:
                    m_p = copy.deepcopy(ref_m)
                    o_p = make_optimizer(arm_probe, m_p.parameters(), lr=3.5e-3)
                    for p_src, p_dst in zip(ref_m.parameters(), m_p.parameters()):
                        st_s = ref_o.state[p_src]
                        o_p.state[p_dst] = {
                            "step": int(st_s["step"]),
                            "exp_avg": st_s["exp_avg"].clone(),
                            "exp_avg_sq": st_s["exp_avg_sq"].clone(),
                        }
                        p_dst.grad = p_src.grad.clone()
                    w_before = [p.detach().clone() for p in m_p.parameters()]
                    o_p.step()
                    dtheta_sq = sum(
                        float((p.detach() - wb).float().pow(2).sum().item())
                        for p, wb in zip(m_p.parameters(), w_before)
                    )
                    loss_after1 = eval_fast_test(m_p)
                    step_resp["arms"][arm_probe] = {
                        "delta_theta_l2": round(math.sqrt(dtheta_sq), 4),
                        "delta_test_nll_1step": round(loss_after1 - loss_before, 5),
                    }
                    del m_p, o_p
                shock_micro_responses.append(step_resp)

        ref_o.step()

    throughput = {}
    for arm_tp in [
        "C01_AdamW",
        "B16_GradClip",
        "B17_StableAdamW_Exact",
        "N20_ExactTAdamW",
        "C09_FULD_raw",
        "T13_AdaFULD_PriorFloor",
    ]:
        m_tp = copy.deepcopy(base_m)
        o_tp = make_optimizer(arm_tp, m_tp.parameters(), lr=3.5e-3)
        xb, yb = _slice_batch(splits["lit_train"], 0)
        for _ in range(15):
            o_tp.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, l = m_tp(xb, yb)
            l.backward()
            o_tp.step()
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        for _ in range(150):
            o_tp.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, l = m_tp(xb, yb)
            l.backward()
            o_tp.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start
        ms_step = 1000.0 * elapsed / 150.0
        tok_s = (150 * BATCH_SIZE * SEQ_LEN) / elapsed
        throughput[arm_tp] = {"ms_per_step": round(ms_step, 3), "tokens_per_sec": round(tok_s, 1)}
        del m_tp, o_tp

    return {
        "manifest": manifest,
        "exceedance_and_energy": {
            "shock_hit_tensors_exceed_pct_mean": round(float(np.mean(hit_exceed_t13)), 3),
            "clean_unhit_tensors_exceed_pct_mean": round(float(np.mean(unhit_exceed_t13)), 4),
            "shock_hit_energy_gap_mean": round(float(np.mean(hit_energy_gap)), 6),
            "clean_unhit_energy_gap_mean": round(float(np.mean(unhit_energy_gap)), 8),
            "unseen_lit_rows_exceed_pct_no_prior_floor_first25": round(float(np.mean(unseen_exceed_no_floor)), 2),
            "unseen_lit_rows_exceed_pct_with_prior_floor_first25": round(float(np.mean(unseen_exceed_with_floor)), 3),
        },
        "shock_1step_micro_responses": shock_micro_responses,
        "gpu_throughput_a10g": throughput,
    }


@app.function(gpu="A10G", timeout=1800)
def run_dev_lr_cell(seed: int, reg_id: str) -> dict:
    """Tune 4-LR grid strictly on Dev Books for a given tuning seed and regime."""
    import sys

    sys.path.insert(0, "/root")

    import torch
    from adafuld import make_optimizer

    device = torch.device("cuda")
    splits, manifest = _prepare_3way_corpora(device)
    MiniGPT = _build_model_cls()
    vocab_size = manifest["vocab_size"]
    span32 = BATCH_SIZE * (SEQ_LEN + 1)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(seed)
    base_m = MiniGPT(vocab_size).to(device)
    base_o = make_optimizer("C01_AdamW", base_m.parameters(), lr=5e-3)
    for s in range(600):
        base_o.param_groups[0]["lr"] = 5e-3 * min(1.0, (s + 1) / 50.0)
        xb, yb = _slice_batch(splits["code_train"], s * span32)
        base_o.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = base_m(xb, yb)
        loss.backward()
        base_o.step()

    @torch.no_grad()
    def eval_dev(m) -> float:
        m.eval()
        tot = 0.0
        for b in range(EVAL_BATCHES):
            xb, yb = _slice_batch(splits["lit_dev"], b * span32)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, l = m(xb, yb)
            tot += float(l.item())
        m.train()
        v = tot / EVAL_BATCHES
        return v if math.isfinite(v) else 99.0

    light_sched, heavy_sched = _build_paired_shock_schedules(seed, 350, len(list(base_m.parameters())))
    sched = None if reg_id == "R_clean" else (light_sched if reg_id == "R_light_gauss" else heavy_sched)

    dev_scores: dict[str, dict[str, float]] = {}
    for method in TUNED_METHODS:
        dev_scores[method] = {}
        for lr_cand in LR_GRID:
            m_c = copy.deepcopy(base_m)
            o_c = make_optimizer(method, m_c.parameters(), lr=lr_cand)
            for p_b, p_t in zip(base_m.parameters(), m_c.parameters()):
                st_b = base_o.state[p_b]
                o_c.state[p_t] = {
                    "step": int(st_b["step"]),
                    "exp_avg": st_b["exp_avg"].clone(),
                    "exp_avg_sq": st_b["exp_avg_sq"].clone(),
                }
            for s in range(350):
                xb, yb = _slice_batch(splits["lit_train"], s * span32)
                o_c.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _, l = m_c(xb, yb)
                l.backward()
                ev = sched[s] if sched is not None else None
                if ev is not None:
                    g_main = [p.grad.detach().clone() for p in m_c.parameters()]
                    xb_a, yb_a = _slice_batch(splits["lit_train"], (420 + s) * span32)
                    o_c.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        _, l_a = m_c(xb_a, yb_a)
                    l_a.backward()
                    for p, gm, xi in zip(m_c.parameters(), g_main, ev):
                        if xi != 0.0:
                            p.grad.copy_(gm + xi * (p.grad.detach() - gm))
                        else:
                            p.grad.copy_(gm)
                o_c.step()
            dev_scores[method][str(lr_cand)] = round(eval_dev(m_c), 4)
            del m_c, o_c

    print(f"[Dev Tune] Completed seed={seed}, reg={reg_id}", flush=True)
    return {"seed": seed, "reg_id": reg_id, "dev_scores": dev_scores}


@app.function(gpu="A10G", timeout=1800)
def run_test_seed(seed: int, best_lrs_json: str) -> dict:
    """Evaluate all Dev-tuned methods + fixed-LR mechanistic isolations on Held-Out Test Books."""
    import sys

    sys.path.insert(0, "/root")

    import numpy as np
    import torch
    import torch.nn.functional as F
    from adafuld import make_optimizer

    best_lrs = json.loads(best_lrs_json)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    splits, manifest = _prepare_3way_corpora(device)
    MiniGPT = _build_model_cls()
    vocab_size = manifest["vocab_size"]
    unk_id = manifest["unk_id"]
    span32 = BATCH_SIZE * (SEQ_LEN + 1)

    torch.manual_seed(seed)
    base_m = MiniGPT(vocab_size).to(device)
    base_o = make_optimizer("C01_AdamW", base_m.parameters(), lr=5e-3)
    for s in range(600):
        base_o.param_groups[0]["lr"] = 5e-3 * min(1.0, (s + 1) / 50.0)
        xb, yb = _slice_batch(splits["code_train"], s * span32)
        base_o.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = base_m(xb, yb)
        loss.backward()
        base_o.step()

    code_prefix_counts = torch.bincount(splits["code_train"][: 600 * span32].cpu(), minlength=vocab_size).to(device)
    lit_train_counts = torch.bincount(splits["lit_train"][: 400 * span32].cpu(), minlength=vocab_size).to(device)

    @torch.no_grad()
    def eval_test_full(m) -> dict:
        m.eval()
        tot_all = 0.0
        known_sum, known_cnt = 0.0, 0
        unseen_sum, unseen_cnt = 0.0, 0
        rare_sum, rare_cnt = 0.0, 0
        freq_sum, freq_cnt = 0.0, 0
        for b in range(EVAL_BATCHES):
            xb, yb = _slice_batch(splits["lit_test"], b * span32)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = m(xb, None)
                tok_l = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), reduction="none").float()
            tot_all += float(tok_l.mean().item())
            yf = yb.view(-1)
            kn = yf != unk_id
            known_sum += float(tok_l[kn].sum().item())
            known_cnt += int(kn.sum().item())
            mu = kn & (code_prefix_counts[yf] == 0)
            mr = kn & (lit_train_counts[yf] <= 30)
            mf = kn & (lit_train_counts[yf] > 200)
            if mu.any():
                unseen_sum += float(tok_l[mu].sum().item())
                unseen_cnt += int(mu.sum().item())
            if mr.any():
                rare_sum += float(tok_l[mr].sum().item())
                rare_cnt += int(mr.sum().item())
            if mf.any():
                freq_sum += float(tok_l[mf].sum().item())
                freq_cnt += int(mf.sum().item())

        code_tot = 0.0
        for b in range(16):
            xb, yb = _slice_batch(splits["code_test"], b * span32)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, lc = m(xb, yb)
            code_tot += float(lc.item())
        m.train()
        all_nll = tot_all / EVAL_BATCHES
        if not math.isfinite(all_nll):
            return {
                "test_nll_all": 99.0,
                "test_nll_known": 99.0,
                "rare_lit_nll": 99.0,
                "unseen_code_nll": 99.0,
                "freq_lit_nll": 99.0,
                "code_ret_nll": 99.0,
            }
        return {
            "test_nll_all": round(all_nll, 4),
            "test_nll_known": round(known_sum / max(1, known_cnt), 4),
            "rare_lit_nll": round(rare_sum / max(1, rare_cnt), 4),
            "unseen_code_nll": round(unseen_sum / max(1, unseen_cnt), 4),
            "freq_lit_nll": round(freq_sum / max(1, freq_cnt), 4),
            "code_ret_nll": round(code_tot / 16.0, 4),
        }

    @torch.no_grad()
    def eval_test_scalar(m) -> float:
        m.eval()
        tot = 0.0
        for b in range(EVAL_BATCHES):
            xb, yb = _slice_batch(splits["lit_test"], b * span32)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, l = m(xb, yb)
            tot += float(l.item())
        m.train()
        v = tot / EVAL_BATCHES
        return round(v, 4) if math.isfinite(v) else 99.0

    light_sched, heavy_sched = _build_paired_shock_schedules(seed, 400, len(list(base_m.parameters())))
    start_test_nll = eval_test_scalar(base_m)

    out_seed = {"seed": seed, "start_test_nll": start_test_nll, "tuned_eval": {}, "fixed_lr_2p5e3_isolation": {}}

    for reg_id, sched in [("R_clean", None), ("R_light_gauss", light_sched), ("R_heavy_sas", heavy_sched)]:
        out_seed["tuned_eval"][reg_id] = {}
        for method in TUNED_METHODS:
            lr_star = float(best_lrs[reg_id][method])
            m_a = copy.deepcopy(base_m)
            o_a = make_optimizer(method, m_a.parameters(), lr=lr_star)
            for p_b, p_t in zip(base_m.parameters(), m_a.parameters()):
                st_b = base_o.state[p_b]
                o_a.state[p_t] = {
                    "step": int(st_b["step"]),
                    "exp_avg": st_b["exp_avg"].clone(),
                    "exp_avg_sq": st_b["exp_avg_sq"].clone(),
                }
            curve_steps = [0]
            curve_nll = [start_test_nll]
            for s in range(400):
                xb, yb = _slice_batch(splits["lit_train"], s * span32)
                o_a.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _, l = m_a(xb, yb)
                l.backward()
                ev = sched[s] if sched is not None else None
                if ev is not None:
                    g_main = [p.grad.detach().clone() for p in m_a.parameters()]
                    xb_a, yb_a = _slice_batch(splits["lit_train"], (420 + s) * span32)
                    o_a.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        _, l_a = m_a(xb_a, yb_a)
                    l_a.backward()
                    for p, gm, xi in zip(m_a.parameters(), g_main, ev):
                        if xi != 0.0:
                            p.grad.copy_(gm + xi * (p.grad.detach() - gm))
                        else:
                            p.grad.copy_(gm)
                o_a.step()
                if (s + 1) in (25, 50, 100, 200, 300, 400):
                    curve_steps.append(s + 1)
                    curve_nll.append(eval_test_scalar(m_a))

            metrics = eval_test_full(m_a)
            metrics["selected_lr"] = lr_star
            metrics["auc_400"] = round(float(np.trapezoid(curve_nll, curve_steps) / 400.0), 4)
            metrics["curve_steps"] = curve_steps
            metrics["curve_nll"] = curve_nll
            out_seed["tuned_eval"][reg_id][method] = metrics
            del m_a, o_a

        out_seed["fixed_lr_2p5e3_isolation"][reg_id] = {}
        for iso_arm in [
            "C01_AdamW",
            "B18_LargeEps_1e-4",
            "C11_FULD_cap_noFloor",
            "T13_FlatTop",
            "T13_AdaFULD_PriorFloor",
            "T13_Plus_MildEps",
        ]:
            m_i = copy.deepcopy(base_m)
            o_i = make_optimizer(iso_arm, m_i.parameters(), lr=2.5e-3)
            for p_b, p_t in zip(base_m.parameters(), m_i.parameters()):
                st_b = base_o.state[p_b]
                o_i.state[p_t] = {
                    "step": int(st_b["step"]),
                    "exp_avg": st_b["exp_avg"].clone(),
                    "exp_avg_sq": st_b["exp_avg_sq"].clone(),
                }
            for s in range(400):
                xb, yb = _slice_batch(splits["lit_train"], s * span32)
                o_i.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _, l = m_i(xb, yb)
                l.backward()
                ev = sched[s] if sched is not None else None
                if ev is not None:
                    g_main = [p.grad.detach().clone() for p in m_i.parameters()]
                    xb_a, yb_a = _slice_batch(splits["lit_train"], (420 + s) * span32)
                    o_i.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        _, l_a = m_i(xb_a, yb_a)
                    l_a.backward()
                    for p, gm, xi in zip(m_i.parameters(), g_main, ev):
                        if xi != 0.0:
                            p.grad.copy_(gm + xi * (p.grad.detach() - gm))
                        else:
                            p.grad.copy_(gm)
                o_i.step()
            out_seed["fixed_lr_2p5e3_isolation"][reg_id][iso_arm] = eval_test_full(m_i)
            del m_i, o_i

        print(f"[Test Seed {seed}] Completed {reg_id}", flush=True)

    return out_seed


@app.local_entrypoint()
def main():
    import numpy as np
    from scipy import stats

    t0 = time.perf_counter()
    print("Stage 1: Spawning Step-Telemetry Worker + 6 Parallel Dev LR-Tuning Workers...")
    telem_handle = run_step_telemetry.spawn()
    dev_jobs = [(sd, rg) for sd in DEV_SEEDS for rg in ["R_clean", "R_light_gauss", "R_heavy_sas"]]
    dev_outs = list(run_dev_lr_cell.starmap(dev_jobs))

    dev_grid_table: dict[str, dict[str, dict[str, float]]] = {}
    best_lrs: dict[str, dict[str, float]] = {}
    for rg in ["R_clean", "R_light_gauss", "R_heavy_sas"]:
        dev_grid_table[rg] = {}
        best_lrs[rg] = {}
        sub = [x["dev_scores"] for x in dev_outs if x["reg_id"] == rg]
        for m in TUNED_METHODS:
            dev_grid_table[rg][m] = {}
            best_val, best_lr = 1e9, LR_GRID[0]
            for lr_c in LR_GRID:
                k = str(lr_c)
                mean_dev = float(np.mean([s[m][k] for s in sub]))
                dev_grid_table[rg][m][k] = round(mean_dev, 4)
                if mean_dev < best_val:
                    best_val = mean_dev
                    best_lr = lr_c
            best_lrs[rg][m] = best_lr

    print("Selected Dev-Optimal LRs:", json.dumps(best_lrs, indent=2), flush=True)

    print("Stage 2: Launching 8 Parallel Fresh Test Seeds [101..808] on Held-Out Test Books...")
    best_lrs_str = json.dumps(best_lrs)
    test_outs = list(run_test_seed.starmap([(sd, best_lrs_str) for sd in TEST_SEEDS]))
    telem_res = telem_handle.get()

    def summarize_group(group_key: str, arm_list: list[str]) -> dict:
        res = {}
        for rg in ["R_clean", "R_light_gauss", "R_heavy_sas"]:
            res[rg] = {}
            for arm in arm_list:
                recs = [sr[group_key][rg][arm] for sr in test_outs]
                scalar_keys = [k for k in recs[0].keys() if isinstance(recs[0][k], (int, float))]
                st = {}
                for k in scalar_keys:
                    arr = np.array([r[k] for r in recs], dtype=np.float64)
                    st[f"{k}_mean"] = round(float(np.mean(arr)), 4)
                    st[f"{k}_sd"] = round(float(np.std(arr, ddof=1)), 4)
                    st[f"{k}_per_seed"] = [round(float(v), 4) for v in arr]
                res[rg][arm] = st
        return res

    tuned_summary = summarize_group("tuned_eval", TUNED_METHODS)
    iso_arms = [
        "C01_AdamW",
        "B18_LargeEps_1e-4",
        "C11_FULD_cap_noFloor",
        "T13_FlatTop",
        "T13_AdaFULD_PriorFloor",
        "T13_Plus_MildEps",
    ]
    iso_summary = summarize_group("fixed_lr_2p5e3_isolation", iso_arms)

    PRIMARY_TESTS = [
        (
            "H1_T13_vs_AdamW_HeavyShock_Tuned",
            "tuned_eval",
            "R_heavy_sas",
            "test_nll_all",
            "T13_AdaFULD_PriorFloor",
            "C01_AdamW",
            "T13 beats LR-tuned AdamW on Held-Out Test Books under Heavy-Tailed Shocks",
        ),
        (
            "H2_T13_vs_StableAdamW_HeavyShock_Tuned",
            "tuned_eval",
            "R_heavy_sas",
            "test_nll_all",
            "T13_AdaFULD_PriorFloor",
            "B17_StableAdamW_Exact",
            "T13 beats LR-tuned Exact StableAdamW under Heavy-Tailed Shocks",
        ),
        (
            "H3_T13_vs_GradClip_HeavyShock_Tuned",
            "tuned_eval",
            "R_heavy_sas",
            "test_nll_all",
            "T13_AdaFULD_PriorFloor",
            "B16_GradClip",
            "T13 beats LR-tuned Global GradClip under Localized Heavy-Tailed Shocks",
        ),
        (
            "H4_T13_vs_LargeEps_RareTokens_Clean_Tuned",
            "tuned_eval",
            "R_clean",
            "rare_lit_nll",
            "T13_AdaFULD_PriorFloor",
            "B18_LargeEps_1e-4",
            "T13 avoids LargeEps rare-token penalty on Held-Out Test Books",
        ),
        (
            "H5_PriorFloor_Rescue_T13_vs_C11_HeavyShock_Iso",
            "fixed_lr_2p5e3_isolation",
            "R_heavy_sas",
            "test_nll_all",
            "T13_AdaFULD_PriorFloor",
            "C11_FULD_cap_noFloor",
            "Prior floor rho_prior=0.15 rescues FULD+VCap from awakening trap (Prop. 2B)",
        ),
        (
            "H6_Heavy_vs_Light_Advantage_Expansion",
            "DIFF_IN_DIFF",
            "R_heavy_vs_light",
            "test_nll_all",
            "T13_AdaFULD_PriorFloor",
            "C01_AdamW",
            "T13 gain over AdamW under Heavy-Tailed shocks exceeds gain under Light-Tailed Gaussian shocks",
        ),
    ]

    raw_pvals = []
    primary_results = {}
    for tid, grp, rg, met, a1, a2, desc in PRIMARY_TESTS:
        if grp == "DIFF_IN_DIFF":
            d_heavy = np.array(
                [sr["tuned_eval"]["R_heavy_sas"][a1][met] - sr["tuned_eval"]["R_heavy_sas"][a2][met] for sr in test_outs]
            )
            d_light = np.array(
                [
                    sr["tuned_eval"]["R_light_gauss"][a1][met] - sr["tuned_eval"]["R_light_gauss"][a2][met]
                    for sr in test_outs
                ]
            )
            diffs = d_heavy - d_light
        else:
            v1 = np.array([sr[grp][rg][a1][met] for sr in test_outs])
            v2 = np.array([sr[grp][rg][a2][met] for sr in test_outs])
            diffs = v1 - v2

        wins = int(np.sum(diffs < 0.0))
        t_res = stats.ttest_rel(diffs, np.zeros_like(diffs))
        s_res = stats.binomtest(wins, n=len(TEST_SEEDS), p=0.5, alternative="two-sided")
        p_t = float(t_res.pvalue)
        raw_pvals.append((tid, p_t))
        primary_results[tid] = {
            "desc": desc,
            "mean_diff": round(float(np.mean(diffs)), 4),
            "sd_diff": round(float(np.std(diffs, ddof=1)), 4),
            "wins_out_of_8": f"{wins}/8",
            "paired_t_p_raw": round(p_t, 6),
            "exact_sign_p_raw": round(float(s_res.pvalue), 5),
            "per_seed_diffs": [round(float(x), 4) for x in diffs],
        }

    sorted_p = sorted(raw_pvals, key=lambda x: x[1])
    m_tot = len(sorted_p)
    running_max = 0.0
    for rank_idx, (tid, p_val) in enumerate(sorted_p):
        adj = min(1.0, p_val * (m_tot - rank_idx))
        running_max = max(running_max, adj)
        primary_results[tid]["holm_bonferroni_p"] = round(running_max, 6)

    EXPLORATORY = [
        ("E1_T13_vs_T13FlatTop_Heavy_Tuned", "tuned_eval", "R_heavy_sas", "test_nll_all", "T13_AdaFULD_PriorFloor", "T13_FlatTop"),
        ("E2_T13_vs_T13FlatTop_Heavy_Iso", "fixed_lr_2p5e3_isolation", "R_heavy_sas", "test_nll_all", "T13_AdaFULD_PriorFloor", "T13_FlatTop"),
        ("E3_T13_vs_ExactTAdamW_Heavy_Tuned", "tuned_eval", "R_heavy_sas", "test_nll_all", "T13_AdaFULD_PriorFloor", "N20_ExactTAdamW"),
        ("E4_T13_vs_CoordClip_Heavy_Tuned", "tuned_eval", "R_heavy_sas", "test_nll_all", "T13_AdaFULD_PriorFloor", "N19_CoordClip"),
        ("E5_T13_vs_C09_RareTokens_Heavy_Tuned", "tuned_eval", "R_heavy_sas", "rare_lit_nll", "T13_AdaFULD_PriorFloor", "C09_FULD_raw"),
        ("E6_T13_vs_C09_Overall_Heavy_Tuned", "tuned_eval", "R_heavy_sas", "test_nll_all", "T13_AdaFULD_PriorFloor", "C09_FULD_raw"),
        ("E7_T13PlusMildEps_vs_T13_Clean_Iso", "fixed_lr_2p5e3_isolation", "R_clean", "test_nll_all", "T13_Plus_MildEps", "T13_AdaFULD_PriorFloor"),
    ]
    exploratory_results = {}
    for eid, grp, rg, met, a1, a2 in EXPLORATORY:
        diffs = np.array([sr[grp][rg][a1][met] - sr[grp][rg][a2][met] for sr in test_outs])
        wins = int(np.sum(diffs < 0.0))
        p_t = float(stats.ttest_rel(diffs, np.zeros_like(diffs)).pvalue)
        exploratory_results[eid] = {
            "mean_diff": round(float(np.mean(diffs)), 4),
            "sd_diff": round(float(np.std(diffs, ddof=1)), 4),
            "wins_out_of_8": f"{wins}/8",
            "paired_t_p_nominal": round(p_t, 5),
            "per_seed_diffs": [round(float(x), 4) for x in diffs],
        }

    final_payload = {
        "elapsed_wall_sec": round(time.perf_counter() - t0, 1),
        "dev_seeds": DEV_SEEDS,
        "test_seeds": TEST_SEEDS,
        "lr_grid": LR_GRID,
        "step_telemetry_and_throughput": telem_res,
        "dev_lr_grid_scores": dev_grid_table,
        "selected_dev_optimal_lrs": best_lrs,
        "test_tuned_summary_8seeds": tuned_summary,
        "test_fixed_lr_2p5e3_isolation_8seeds": iso_summary,
        "preregistered_primary_tests_holm": primary_results,
        "exploratory_comparisons": exploratory_results,
        "raw_test_seed_runs": test_outs,
    }
    out_path = REPO_ROOT / "results" / "exp_results_v3.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(final_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved complete benchmark results to {out_path} in {final_payload['elapsed_wall_sec']}s")

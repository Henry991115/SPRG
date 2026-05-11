"""Contrastive decoding loops for InternVL (LLaVA-style formulas, no HF LogitsProcessor)."""

from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.nn.functional as F

from .internvl_chat_utils import (
    encode_user_body_to_input_ids,
    internvl_forward_vcd_decode_step,
    internvl_forward_vcd_prefill,
    sample_next_token,
)


def _jsd_pq(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum()
    kl_qm = (q * (q.log() - m.log())).sum()
    return 0.5 * (kl_pm + kl_qm)


def _normalized_entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    k = p.numel()
    if k <= 1:
        return p.new_zeros(())
    h = -(p * (p + eps).log()).sum()
    return h / torch.log(torch.tensor(float(k), device=p.device, dtype=p.dtype))


def generate_standard_chat(
    model: torch.nn.Module,
    tokenizer: Any,
    pixel_values: torch.Tensor,
    question: str,
    max_new_tokens: int,
    temperature: float,
    top_p: Optional[float],
) -> str:
    gen_cfg = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
    )
    if top_p is not None:
        gen_cfg["top_p"] = top_p
    q = question if "<image>" in question else f"<image>\n{question}"
    return model.chat(tokenizer, pixel_values, q, gen_cfg).strip()


def generate_vcd(
    model: torch.nn.Module,
    tokenizer: Any,
    expert_pv: torch.Tensor,
    amateur_pv: torch.Tensor,
    user_body: str,
    device: torch.device,
    cad_alpha: float = 0.2,
    max_new_tokens: int = 1024,
    temperature: float = 0.2,
    top_p: Optional[float] = None,
) -> str:
    """VCD: logits_E - cad_alpha * logits_A; dual KV prefill then incremental decode without ViT."""
    num_patches = int(expert_pv.shape[0])
    input_ids, attention_mask, eos_id = encode_user_body_to_input_ids(
        model, tokenizer, user_body, num_patches, device
    )
    gen_ids: List[int] = []
    curr_attn = attention_mask.clone()
    past_e = past_a = None
    id_dtype = input_ids.dtype

    for _ in range(max_new_tokens):
        if past_e is None:
            le, past_e = internvl_forward_vcd_prefill(
                model, expert_pv, input_ids, curr_attn
            )
            la, past_a = internvl_forward_vcd_prefill(
                model, amateur_pv, input_ids, curr_attn
            )
        else:
            nt = torch.tensor([[gen_ids[-1]]], device=device, dtype=id_dtype)
            curr_attn = torch.cat(
                [
                    curr_attn,
                    torch.ones(1, 1, dtype=curr_attn.dtype, device=device),
                ],
                dim=1,
            )
            le, past_e = internvl_forward_vcd_decode_step(
                model, nt, curr_attn, past_e
            )
            la, past_a = internvl_forward_vcd_decode_step(
                model, nt, curr_attn, past_a
            )

        logits = le - float(cad_alpha) * la
        next_id = sample_next_token(logits, temperature, top_p)
        if next_id == eos_id:
            break
        gen_ids.append(next_id)

    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def generate_joint_icd_vcd(
    model: torch.nn.Module,
    tokenizer: Any,
    expert_pv: torch.Tensor,
    vcd_pv: torch.Tensor,
    icd_pv: torch.Tensor,
    expert_user_body: str,
    icd_prefix_ids: torch.Tensor,
    expert_prompt_len: int,
    icd_prompt_len: int,
    device: torch.device,
    alpha: float,
    beta: float,
    max_new_tokens: int = 1024,
    temperature: float = 0.2,
    top_p: Optional[float] = None,
) -> str:
    """Joint ICD+VCD: logits = le - alpha * relu(lv - eta*li) - beta * li; shared expert/VCD prompt."""
    _ = expert_prompt_len
    eta = 0.5
    num_patches = int(expert_pv.shape[0])
    curr_ids, attn_ev, eos_id = encode_user_body_to_input_ids(
        model, tokenizer, expert_user_body, num_patches, device
    )
    icd_ids_init = icd_prefix_ids[:, :icd_prompt_len]
    icd_attn = torch.ones_like(icd_ids_init, dtype=torch.long, device=device)
    gen_ids: List[int] = []
    past_e = past_v = past_i = None
    id_dtype = curr_ids.dtype

    for _ in range(max_new_tokens):
        if past_e is None:
            le, past_e = internvl_forward_vcd_prefill(
                model, expert_pv, curr_ids, attn_ev
            )
            lv, past_v = internvl_forward_vcd_prefill(
                model, vcd_pv, curr_ids, attn_ev
            )
            li, past_i = internvl_forward_vcd_prefill(
                model, icd_pv, icd_ids_init, icd_attn
            )
        else:
            nt = torch.tensor([[gen_ids[-1]]], device=device, dtype=id_dtype)
            attn_ev = torch.cat(
                [attn_ev, torch.ones(1, 1, dtype=attn_ev.dtype, device=device)], dim=1
            )
            icd_attn = torch.cat(
                [icd_attn, torch.ones(1, 1, dtype=icd_attn.dtype, device=device)], dim=1
            )
            le, past_e = internvl_forward_vcd_decode_step(
                model, nt, attn_ev, past_e
            )
            lv, past_v = internvl_forward_vcd_decode_step(
                model, nt, attn_ev, past_v
            )
            li, past_i = internvl_forward_vcd_decode_step(
                model, nt, icd_attn, past_i
            )
        lv_res = F.relu(lv - eta * li)
        logits = le - float(alpha) * lv_res - float(beta) * li
        next_id = sample_next_token(logits, temperature, top_p)
        if next_id == eos_id:
            break
        gen_ids.append(next_id)
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def generate_joint_icd_vcd_split(
    model: torch.nn.Module,
    tokenizer: Any,
    expert_pv: torch.Tensor,
    vcd_pv: torch.Tensor,
    icd_pv: torch.Tensor,
    expert_user_body: str,
    vcd_user_body: str,
    icd_user_body: str,
    device: torch.device,
    alpha: float,
    beta: float,
    max_new_tokens: int = 1024,
    temperature: float = 0.2,
    top_p: Optional[float] = None,
) -> str:
    """Same joint objective as generate_joint_icd_vcd but separate chat templates per branch."""
    eta = 0.5
    num_patches = int(expert_pv.shape[0])
    curr_e, attn_e, eos_id = encode_user_body_to_input_ids(
        model, tokenizer, expert_user_body, num_patches, device
    )
    curr_v, attn_v, eos_v = encode_user_body_to_input_ids(
        model, tokenizer, vcd_user_body, num_patches, device
    )
    icd_prefix_ids, _, _ = encode_user_body_to_input_ids(
        model, tokenizer, icd_user_body, num_patches, device
    )
    if int(eos_v) != int(eos_id):
        raise RuntimeError("split joint: expert/vcd eos mismatch")
    icd_prompt_len = int(icd_prefix_ids.shape[1])
    icd_ids_init = icd_prefix_ids[:, :icd_prompt_len]
    icd_attn = torch.ones_like(icd_ids_init, dtype=torch.long, device=device)
    gen_ids: List[int] = []
    past_e = past_v = past_i = None
    id_dtype = curr_e.dtype

    for _ in range(max_new_tokens):
        if past_e is None:
            le, past_e = internvl_forward_vcd_prefill(
                model, expert_pv, curr_e, attn_e
            )
            lv, past_v = internvl_forward_vcd_prefill(
                model, vcd_pv, curr_v, attn_v
            )
            li, past_i = internvl_forward_vcd_prefill(
                model, icd_pv, icd_ids_init, icd_attn
            )
        else:
            nt = torch.tensor([[gen_ids[-1]]], device=device, dtype=id_dtype)
            attn_e = torch.cat(
                [attn_e, torch.ones(1, 1, dtype=attn_e.dtype, device=device)], dim=1
            )
            attn_v = torch.cat(
                [attn_v, torch.ones(1, 1, dtype=attn_v.dtype, device=device)], dim=1
            )
            icd_attn = torch.cat(
                [icd_attn, torch.ones(1, 1, dtype=icd_attn.dtype, device=device)], dim=1
            )
            le, past_e = internvl_forward_vcd_decode_step(
                model, nt, attn_e, past_e
            )
            lv, past_v = internvl_forward_vcd_decode_step(
                model, nt, attn_v, past_v
            )
            li, past_i = internvl_forward_vcd_decode_step(
                model, nt, icd_attn, past_i
            )
        lv_res = F.relu(lv - eta * li)
        logits = le - float(alpha) * lv_res - float(beta) * li
        next_id = sample_next_token(logits, temperature, top_p)
        if next_id == eos_id:
            break
        gen_ids.append(next_id)
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def generate_adaptive_joint_icd_vcd(
    model: torch.nn.Module,
    tokenizer: Any,
    expert_pv: torch.Tensor,
    vcd_pv: torch.Tensor,
    icd_pv: torch.Tensor,
    expert_user_body: str,
    icd_prefix_ids: torch.Tensor,
    expert_prompt_len: int,
    icd_prompt_len: int,
    device: torch.device,
    top_k: int = 50,
    lambda_min: float = 0.0,
    lambda_max: float = 1.0,
    tau: float = 1.0,
    icd_bias: float = 0.0,
    vcd_residual_eta: float = 0.5,
    max_new_tokens: int = 1024,
    temperature: float = 0.2,
    top_p: Optional[float] = None,
) -> str:
    """Adaptive alpha_t/beta_t from expert top-k JSD and normalized entropy."""
    _ = expert_prompt_len
    num_patches = int(expert_pv.shape[0])
    curr_ids, attn_ev, eos_id = encode_user_body_to_input_ids(
        model, tokenizer, expert_user_body, num_patches, device
    )
    icd_ids_init = icd_prefix_ids[:, :icd_prompt_len]
    icd_attn = torch.ones_like(icd_ids_init, dtype=torch.long, device=device)
    gen_ids: List[int] = []
    past_e = past_v = past_i = None
    id_dtype = curr_ids.dtype

    for _ in range(max_new_tokens):
        if past_e is None:
            le, past_e = internvl_forward_vcd_prefill(
                model, expert_pv, curr_ids, attn_ev
            )
            lv, past_v = internvl_forward_vcd_prefill(
                model, vcd_pv, curr_ids, attn_ev
            )
            li, past_i = internvl_forward_vcd_prefill(
                model, icd_pv, icd_ids_init, icd_attn
            )
        else:
            nt = torch.tensor([[gen_ids[-1]]], device=device, dtype=id_dtype)
            attn_ev = torch.cat(
                [attn_ev, torch.ones(1, 1, dtype=attn_ev.dtype, device=device)], dim=1
            )
            icd_attn = torch.cat(
                [icd_attn, torch.ones(1, 1, dtype=icd_attn.dtype, device=device)], dim=1
            )
            le, past_e = internvl_forward_vcd_decode_step(
                model, nt, attn_ev, past_e
            )
            lv, past_v = internvl_forward_vcd_decode_step(
                model, nt, attn_ev, past_v
            )
            li, past_i = internvl_forward_vcd_decode_step(
                model, nt, icd_attn, past_i
            )

        vocab = le.numel()
        k = max(1, min(int(top_k), vocab))
        _, topk_idx = torch.topk(le, k=k)
        le_k = le[topk_idx]
        lv_k = lv[topk_idx]
        li_k = li[topk_idx]
        le_n = le_k - le_k.max()
        lv_n = lv_k - lv_k.max()
        li_n = li_k - li_k.max()
        p_E = F.softmax(le_n, dim=0)
        p_V = F.softmax(lv_n, dim=0)
        p_I = F.softmax(li_n, dim=0)
        u_t = _normalized_entropy(p_E)
        s_V = 1.0 - _jsd_pq(p_E, p_V)
        s_I = 1.0 - _jsd_pq(p_E, p_I)
        r_V = u_t * s_V
        r_I = u_t * s_I
        lam_budget = lambda_min + (lambda_max - lambda_min) * torch.max(r_V, r_I)
        t = max(float(tau), 1e-8)
        logits_w = torch.stack([r_V, r_I + float(icd_bias)], dim=0) / t
        w = F.softmax(logits_w, dim=0)
        alpha_t = lam_budget * w[0]
        beta_t = lam_budget * w[1]
        lv_res = F.relu(lv - float(vcd_residual_eta) * li)
        logits = le - alpha_t * lv_res - beta_t * li
        next_id = sample_next_token(logits, temperature, top_p)
        if next_id == eos_id:
            break
        gen_ids.append(next_id)
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def generate_adaptive_joint_icd_vcd_split(
    model: torch.nn.Module,
    tokenizer: Any,
    expert_pv: torch.Tensor,
    vcd_pv: torch.Tensor,
    icd_pv: torch.Tensor,
    expert_user_body: str,
    vcd_user_body: str,
    icd_user_body: str,
    device: torch.device,
    top_k: int = 50,
    lambda_min: float = 0.0,
    lambda_max: float = 1.0,
    tau: float = 1.0,
    icd_bias: float = 0.0,
    vcd_residual_eta: float = 0.5,
    max_new_tokens: int = 1024,
    temperature: float = 0.2,
    top_p: Optional[float] = None,
) -> str:
    """Adaptive joint ICD+VCD with split prompts per branch."""
    num_patches = int(expert_pv.shape[0])
    curr_e, attn_e, eos_id = encode_user_body_to_input_ids(
        model, tokenizer, expert_user_body, num_patches, device
    )
    curr_v, attn_v, eos_v = encode_user_body_to_input_ids(
        model, tokenizer, vcd_user_body, num_patches, device
    )
    icd_prefix_ids, _, _ = encode_user_body_to_input_ids(
        model, tokenizer, icd_user_body, num_patches, device
    )
    if int(eos_v) != int(eos_id):
        raise RuntimeError("split joint: expert/vcd eos mismatch")
    icd_prompt_len = int(icd_prefix_ids.shape[1])
    icd_ids_init = icd_prefix_ids[:, :icd_prompt_len]
    icd_attn = torch.ones_like(icd_ids_init, dtype=torch.long, device=device)
    gen_ids: List[int] = []
    past_e = past_v = past_i = None
    id_dtype = curr_e.dtype

    for _ in range(max_new_tokens):
        if past_e is None:
            le, past_e = internvl_forward_vcd_prefill(
                model, expert_pv, curr_e, attn_e
            )
            lv, past_v = internvl_forward_vcd_prefill(
                model, vcd_pv, curr_v, attn_v
            )
            li, past_i = internvl_forward_vcd_prefill(
                model, icd_pv, icd_ids_init, icd_attn
            )
        else:
            nt = torch.tensor([[gen_ids[-1]]], device=device, dtype=id_dtype)
            attn_e = torch.cat(
                [attn_e, torch.ones(1, 1, dtype=attn_e.dtype, device=device)], dim=1
            )
            attn_v = torch.cat(
                [attn_v, torch.ones(1, 1, dtype=attn_v.dtype, device=device)], dim=1
            )
            icd_attn = torch.cat(
                [icd_attn, torch.ones(1, 1, dtype=icd_attn.dtype, device=device)], dim=1
            )
            le, past_e = internvl_forward_vcd_decode_step(
                model, nt, attn_e, past_e
            )
            lv, past_v = internvl_forward_vcd_decode_step(
                model, nt, attn_v, past_v
            )
            li, past_i = internvl_forward_vcd_decode_step(
                model, nt, icd_attn, past_i
            )

        vocab = le.numel()
        k = max(1, min(int(top_k), vocab))
        _, topk_idx = torch.topk(le, k=k)
        le_k = le[topk_idx]
        lv_k = lv[topk_idx]
        li_k = li[topk_idx]
        le_n = le_k - le_k.max()
        lv_n = lv_k - lv_k.max()
        li_n = li_k - li_k.max()
        p_E = F.softmax(le_n, dim=0)
        p_V = F.softmax(lv_n, dim=0)
        p_I = F.softmax(li_n, dim=0)
        u_t = _normalized_entropy(p_E)
        s_V = 1.0 - _jsd_pq(p_E, p_V)
        s_I = 1.0 - _jsd_pq(p_E, p_I)
        r_V = u_t * s_V
        r_I = u_t * s_I
        lam_budget = lambda_min + (lambda_max - lambda_min) * torch.max(r_V, r_I)
        t = max(float(tau), 1e-8)
        logits_w = torch.stack([r_V, r_I + float(icd_bias)], dim=0) / t
        w = F.softmax(logits_w, dim=0)
        alpha_t = lam_budget * w[0]
        beta_t = lam_budget * w[1]
        lv_res = F.relu(lv - float(vcd_residual_eta) * li)
        logits = le - alpha_t * lv_res - beta_t * li
        next_id = sample_next_token(logits, temperature, top_p)
        if next_id == eos_id:
            break
        gen_ids.append(next_id)
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def generate_scalar_top12_joint_icd_vcd(
    model: torch.nn.Module,
    tokenizer: Any,
    expert_pv: torch.Tensor,
    vcd_pv: torch.Tensor,
    icd_pv: torch.Tensor,
    expert_user_body: str,
    icd_prefix_ids: torch.Tensor,
    expert_prompt_len: int,
    icd_prompt_len: int,
    device: torch.device,
    lambda_max: float = 1.0,
    tau: float = 1.0,
    icd_bias: float = 0.0,
    vcd_residual_eta: float = 0.5,
    margin_temperature: float = 1.0,
    max_new_tokens: int = 1024,
    temperature: float = 0.2,
    top_p: Optional[float] = None,
) -> str:
    """Scalar lambda_t from top1-top2 margin; joint ICD+VCD weighting."""
    _ = expert_prompt_len
    num_patches = int(expert_pv.shape[0])
    curr_ids, attn_ev, eos_id = encode_user_body_to_input_ids(
        model, tokenizer, expert_user_body, num_patches, device
    )
    icd_ids_init = icd_prefix_ids[:, :icd_prompt_len]
    icd_attn = torch.ones_like(icd_ids_init, dtype=torch.long, device=device)
    gen_ids: List[int] = []
    past_e = past_v = past_i = None
    id_dtype = curr_ids.dtype

    for _ in range(max_new_tokens):
        if past_e is None:
            le, past_e = internvl_forward_vcd_prefill(
                model, expert_pv, curr_ids, attn_ev
            )
            lv, past_v = internvl_forward_vcd_prefill(
                model, vcd_pv, curr_ids, attn_ev
            )
            li, past_i = internvl_forward_vcd_prefill(
                model, icd_pv, icd_ids_init, icd_attn
            )
        else:
            nt = torch.tensor([[gen_ids[-1]]], device=device, dtype=id_dtype)
            attn_ev = torch.cat(
                [attn_ev, torch.ones(1, 1, dtype=attn_ev.dtype, device=device)], dim=1
            )
            icd_attn = torch.cat(
                [icd_attn, torch.ones(1, 1, dtype=icd_attn.dtype, device=device)], dim=1
            )
            le, past_e = internvl_forward_vcd_decode_step(
                model, nt, attn_ev, past_e
            )
            lv, past_v = internvl_forward_vcd_decode_step(
                model, nt, attn_ev, past_v
            )
            li, past_i = internvl_forward_vcd_decode_step(
                model, nt, icd_attn, past_i
            )

        top2_vals, top2_idx = torch.topk(le, k=min(2, le.numel()))
        y_top1 = int(top2_idx[0].item())
        margin = top2_vals[0] - (
            top2_vals[1] if top2_idx.numel() > 1 else top2_vals[0]
        )
        t_u = max(float(margin_temperature), 1e-8)
        u_t = torch.sigmoid(-margin / t_u)
        lv_res = F.relu(lv - float(vcd_residual_eta) * li)
        r_V = torch.sigmoid(lv_res[y_top1] - le[y_top1])
        r_I = torch.sigmoid(li[y_top1] - le[y_top1])
        lambda_t = float(lambda_max) * u_t
        t = max(float(tau), 1e-8)
        logits_w = torch.stack([r_V, r_I + float(icd_bias)], dim=0) / t
        w = F.softmax(logits_w, dim=0)
        logits = le - (lambda_t * w[0]) * lv_res - (lambda_t * w[1]) * li
        next_id = sample_next_token(logits, temperature, top_p)
        if next_id == eos_id:
            break
        gen_ids.append(next_id)
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def generate_scalar_top12_joint_icd_vcd_split(
    model: torch.nn.Module,
    tokenizer: Any,
    expert_pv: torch.Tensor,
    vcd_pv: torch.Tensor,
    icd_pv: torch.Tensor,
    expert_user_body: str,
    vcd_user_body: str,
    icd_user_body: str,
    device: torch.device,
    lambda_max: float = 1.0,
    tau: float = 1.0,
    icd_bias: float = 0.0,
    vcd_residual_eta: float = 0.5,
    margin_temperature: float = 1.0,
    max_new_tokens: int = 1024,
    temperature: float = 0.2,
    top_p: Optional[float] = None,
) -> str:
    """Scalar margin variant with split prompts per branch."""
    num_patches = int(expert_pv.shape[0])
    curr_e, attn_e, eos_id = encode_user_body_to_input_ids(
        model, tokenizer, expert_user_body, num_patches, device
    )
    curr_v, attn_v, eos_v = encode_user_body_to_input_ids(
        model, tokenizer, vcd_user_body, num_patches, device
    )
    icd_prefix_ids, _, _ = encode_user_body_to_input_ids(
        model, tokenizer, icd_user_body, num_patches, device
    )
    if int(eos_v) != int(eos_id):
        raise RuntimeError("split joint: expert/vcd eos mismatch")
    icd_prompt_len = int(icd_prefix_ids.shape[1])
    icd_ids_init = icd_prefix_ids[:, :icd_prompt_len]
    icd_attn = torch.ones_like(icd_ids_init, dtype=torch.long, device=device)
    gen_ids: List[int] = []
    past_e = past_v = past_i = None
    id_dtype = curr_e.dtype

    for _ in range(max_new_tokens):
        if past_e is None:
            le, past_e = internvl_forward_vcd_prefill(
                model, expert_pv, curr_e, attn_e
            )
            lv, past_v = internvl_forward_vcd_prefill(
                model, vcd_pv, curr_v, attn_v
            )
            li, past_i = internvl_forward_vcd_prefill(
                model, icd_pv, icd_ids_init, icd_attn
            )
        else:
            nt = torch.tensor([[gen_ids[-1]]], device=device, dtype=id_dtype)
            attn_e = torch.cat(
                [attn_e, torch.ones(1, 1, dtype=attn_e.dtype, device=device)], dim=1
            )
            attn_v = torch.cat(
                [attn_v, torch.ones(1, 1, dtype=attn_v.dtype, device=device)], dim=1
            )
            icd_attn = torch.cat(
                [icd_attn, torch.ones(1, 1, dtype=icd_attn.dtype, device=device)], dim=1
            )
            le, past_e = internvl_forward_vcd_decode_step(
                model, nt, attn_e, past_e
            )
            lv, past_v = internvl_forward_vcd_decode_step(
                model, nt, attn_v, past_v
            )
            li, past_i = internvl_forward_vcd_decode_step(
                model, nt, icd_attn, past_i
            )

        top2_vals, top2_idx = torch.topk(le, k=min(2, le.numel()))
        y_top1 = int(top2_idx[0].item())
        margin = top2_vals[0] - (
            top2_vals[1] if top2_idx.numel() > 1 else top2_vals[0]
        )
        t_u = max(float(margin_temperature), 1e-8)
        u_t = torch.sigmoid(-margin / t_u)
        lv_res = F.relu(lv - float(vcd_residual_eta) * li)
        r_V = torch.sigmoid(lv_res[y_top1] - le[y_top1])
        r_I = torch.sigmoid(li[y_top1] - le[y_top1])
        lambda_t = float(lambda_max) * u_t
        t = max(float(tau), 1e-8)
        logits_w = torch.stack([r_V, r_I + float(icd_bias)], dim=0) / t
        w = F.softmax(logits_w, dim=0)
        logits = le - (lambda_t * w[0]) * lv_res - (lambda_t * w[1]) * li
        next_id = sample_next_token(logits, temperature, top_p)
        if next_id == eos_id:
            break
        gen_ids.append(next_id)
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

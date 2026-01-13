import argparse
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from qwenvl.data.rope2d import get_rope_index_3
from qwenvl.train.modeling_qwen3_vl_pointcloud import Qwen3VLModel_PointCloud


def _pick_device(device: str) -> torch.device:
    if device == "cuda":
        return torch.device("cuda")
    if device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pick_dtype(dtype: str) -> Optional[torch.dtype]:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    return None


def _make_text_batch(processor, batch_size: int, max_length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    texts = []
    for i in range(batch_size):
        # Make different lengths to exercise padding.
        rep = 1 + (i % 3)
        texts.append(("hello world " * rep).strip())
    enc = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    if labels.shape[1] > 0:
        labels[:, 0] = -100
    return input_ids, attention_mask, labels


def _check_2d(model, processor, device: torch.device, max_length: int, do_backward: bool) -> None:
    input_ids, attention_mask, labels = _make_text_batch(processor, batch_size=2, max_length=max_length)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    position_ids, _ = get_rope_index_3(attention_mask=attention_mask, input_ids=input_ids)
    position_ids = position_ids.to(device)

    out = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels)
    loss = out.loss
    if loss is None or not torch.isfinite(loss).all():
        raise RuntimeError(f"2D forward produced invalid loss: {loss}")
    if do_backward:
        loss.backward()


def _check_3d(model, processor, device: torch.device, max_length: int, pc_len: int, do_backward: bool) -> None:
    # Patch the inner model forward to the point-aware implementation without copying weights.
    import types as _types

    model.model.forward = _types.MethodType(Qwen3VLModel_PointCloud.forward, model.model)

    input_ids, attention_mask, labels = _make_text_batch(processor, batch_size=2, max_length=max_length)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    hidden_size = int(model.config.text_config.hidden_size)
    point_cloud_embeds = torch.randn((input_ids.shape[0], pc_len, hidden_size), device=device, dtype=model.dtype)

    core_out = model.model(input_ids=input_ids, attention_mask=attention_mask, point_cloud_embeds=point_cloud_embeds)
    hidden_states = core_out.last_hidden_state
    if hidden_states.shape[1] != pc_len + input_ids.shape[1]:
        raise RuntimeError(
            f"3D hidden_states length mismatch: got {hidden_states.shape[1]}, expected {pc_len + input_ids.shape[1]}"
        )

    logits = model.lm_head(hidden_states)
    ignore = labels.new_full((labels.shape[0], pc_len), -100)
    labels_aug = torch.cat([ignore, labels], dim=1)
    shift_labels = labels_aug[:, 1:].contiguous()
    shift_logits = logits[:, :-1, :].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1), ignore_index=-100)
    if not torch.isfinite(loss).all():
        raise RuntimeError(f"3D manual loss is invalid: {loss}")
    if do_backward:
        loss.backward()


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke check Qwen3-VL 2D/3D training paths")
    parser.add_argument("--model", type=str, required=True, help="Local checkpoint path or HF id")
    parser.add_argument("--attn-implementation", type=str, default=None, choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32", "auto"])
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--pc-len", type=int, default=8)
    parser.add_argument("--check-2d", action="store_true")
    parser.add_argument("--check-3d", action="store_true")
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()

    if not args.check_2d and not args.check_3d:
        args.check_2d = True
        args.check_3d = True

    device = _pick_device(args.device)
    torch_dtype = _pick_dtype(args.dtype) if args.dtype != "auto" else None

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        attn_implementation=args.attn_implementation,
        dtype=torch_dtype,
    )
    processor = AutoProcessor.from_pretrained(args.model)
    model.to(device)
    model.train()

    if args.check_2d:
        _check_2d(model, processor, device=device, max_length=int(args.max_length), do_backward=bool(args.backward))
    if args.check_3d:
        _check_3d(
            model,
            processor,
            device=device,
            max_length=int(args.max_length),
            pc_len=int(args.pc_len),
            do_backward=bool(args.backward),
        )

    print("[OK] smoke check passed")


if __name__ == "__main__":
    main()

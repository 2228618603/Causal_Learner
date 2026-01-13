import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
)
from transformers.utils import is_flash_attn_2_available

from qwenvl.data.data_processor import preprocess_qwen_visual
from qwenvl.train.argument import TrainingArguments
from qwenvl.train.model.point_adapter import create_pointcloud_adapter
from qwenvl.train.modeling_qwen3_vl_pointcloud import Qwen3VLModel_PointCloud


try:
    import open3d as o3d  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    o3d = None


def _is_rank0() -> bool:
    return not (torch.distributed.is_available() and torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0


def _unwrap_to_qwen_causal_lm(model: torch.nn.Module) -> torch.nn.Module:
    base_model = getattr(model, "base_model", None)
    if base_model is not None and hasattr(base_model, "model"):
        return base_model.model
    return model


def _get_qwen_core(model: torch.nn.Module) -> torch.nn.Module:
    causal_lm = _unwrap_to_qwen_causal_lm(model)
    return getattr(causal_lm, "model", causal_lm)


class PointCloudVisualDataset(Dataset):
    """点云 +（可选）图像/视频 + 文本 数据集。"""

    def __init__(
        self,
        annotations: List[Dict[str, Any]],
        processor: AutoProcessor,
        pc_adapter_fn,
        pc_token_len: int,
        hidden_size: int,
        use_color: bool = False,
        point_num: Optional[int] = 0,
        embeds_out_device: Optional[torch.device] = None,
    ):
        self.anns = annotations
        self.processor = processor
        self.pc_adapter_fn = pc_adapter_fn
        self.pc_token_len = pc_token_len
        self.hidden_size = hidden_size
        self.use_color = use_color
        self.point_num = point_num
        self.embeds_out_device = embeds_out_device or torch.device("cpu")

    def __len__(self):
        return len(self.anns)

    def _load_points(self, path_or_none: Optional[str], base_path: Optional[Path] = None) -> Optional[torch.Tensor]:
        if not path_or_none:
            return None

        p = Path(path_or_none)
        if not p.is_absolute():
            p = (base_path or Path(".")) / p
        p = p.resolve()

        if not p.exists():
            raise FileNotFoundError(f"Point cloud file not found: {p}")

        if p.suffix.lower() == ".ply":
            if o3d is None:
                raise ModuleNotFoundError(
                    "Reading .ply point clouds requires `open3d`. Install open3d or use .npy/.npz inputs."
                )
            pcd = o3d.io.read_point_cloud(str(p))
            xyz = np.asarray(pcd.points, dtype=np.float32)
            if pcd.has_colors():
                rgb = np.asarray(pcd.colors, dtype=np.float32)
                arr = np.concatenate([xyz, rgb], axis=1)
            else:
                arr = xyz
        else:
            arr = np.load(str(p), allow_pickle=True)
            if isinstance(arr, np.lib.npyio.NpzFile):
                key = list(arr.keys())[0]
                arr = arr[key]

        arr = arr.astype(np.float32)
        if self.use_color:
            if arr.shape[-1] >= 6:
                arr = arr[:, :6]
            elif arr.shape[-1] == 3:
                arr = np.concatenate([arr, np.zeros_like(arr)], axis=1)
            else:
                raise ValueError(f"Unsupported point feature dim={arr.shape[-1]} for use_color=True (expect 3 or >=6)")
        else:
            arr = arr[:, :3] if arr.shape[-1] > 3 else arr

        # Normalize xyz to a unit sphere
        xyz = arr[:, :3]
        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        m = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
        if m > 0:
            xyz = xyz / m

        if self.use_color:
            rgb = arr[:, 3:6]
            if rgb.max() > 1.0:
                rgb = np.clip(rgb / 255.0, 0.0, 1.0)
            arr = np.concatenate([xyz, rgb], axis=1)
        else:
            arr = xyz

        # Optional farthest point sampling
        if isinstance(self.point_num, int) and self.point_num > 0 and arr.shape[0] > self.point_num:
            n = arr.shape[0]
            xyz_np = arr[:, :3]
            centroids = np.zeros((self.point_num,), dtype=np.int64)
            distance = np.ones((n,), dtype=np.float64) * 1e10
            farthest = np.random.randint(0, n)
            for i in range(self.point_num):
                centroids[i] = farthest
                centroid = xyz_np[farthest]
                dist = np.sum((xyz_np - centroid) ** 2, axis=1)
                mask = dist < distance
                distance[mask] = dist[mask]
                farthest = int(np.argmax(distance))
            arr = arr[centroids]

        return torch.from_numpy(arr).float()

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = self.anns[idx]
        if isinstance(sample, list):
            if len(sample) != 1:
                raise ValueError(
                    f"PointCloudVisualDataset expects each annotation entry to be a single sample dict; "
                    f"got list(len={len(sample)}) at idx={idx}."
                )
            sample = sample[0]
        if not isinstance(sample, dict):
            raise TypeError(f"Expected a dict sample at idx={idx}, got {type(sample)}")

        sources = [sample]
        base_path = Path(sample.get("data_path", ""))

        data_dict = preprocess_qwen_visual(sources, self.processor)

        # Position ids (mRoPE) are computed inside the model (HF Qwen3-VL) when `position_ids` is not provided.

        pc_path = sample.get("point_cloud", None)
        pc_tensor = self._load_points(pc_path, base_path=base_path)
        if getattr(self, "use_pc_adapter_forward", False):
            data_dict["point_clouds"] = pc_tensor
        else:
            if pc_tensor is not None:
                pc_embeds = self.pc_adapter_fn(pc_tensor.unsqueeze(0), out_device=self.embeds_out_device).squeeze(0)
            else:
                pc_embeds = torch.zeros(self.pc_token_len, self.hidden_size)
            data_dict["point_cloud_embeds"] = pc_embeds

        return data_dict


class PointCloudCollator:
    """整理 batch：pad 文本 token + 组装视觉张量 + 点云前缀。"""

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.use_pc_adapter_forward: bool = False

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch: Dict[str, torch.Tensor] = {}

        input_ids = [f["input_ids"][0] for f in features]
        batch_tokens = self.tokenizer.pad({"input_ids": input_ids}, return_tensors="pt")
        batch.update({k: v for k, v in batch_tokens.items()})

        max_len = batch["input_ids"].size(1)
        labels_list = [f["labels"][0] for f in features]
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id must be set for padding.")
        padded_labels = []
        for ids, lab in zip(batch["input_ids"], labels_list):
            non_pad = ids.ne(pad_token_id)
            non_pad_len = int(non_pad.long().sum().item())
            if int(lab.numel()) != non_pad_len:
                raise ValueError(
                    f"Label length {int(lab.numel())} does not match non-pad token length {non_pad_len}. "
                    "This usually indicates a padding-side mismatch."
                )
            full = lab.new_full((max_len,), -100)
            full[non_pad] = lab
            padded_labels.append(full)
        batch["labels"] = torch.stack(padded_labels, dim=0)

        if self.use_pc_adapter_forward:
            batch["point_clouds"] = [f.get("point_clouds", None) for f in features]
        else:
            pc_list = [f["point_cloud_embeds"] for f in features]
            batch["point_cloud_embeds"] = torch.stack(pc_list, dim=0)

        img_feats = [(f.get("pixel_values", None), f.get("image_grid_thw", None)) for f in features]
        img_feats = [(pv, thw) for pv, thw in img_feats if pv is not None]
        if img_feats:
            pixel_values_list, thw_list = zip(*img_feats)
            if any(t is None for t in thw_list):
                raise ValueError("`image_grid_thw` is required when `pixel_values` is provided.")
            batch["pixel_values"] = torch.cat(list(pixel_values_list), dim=0)
            batch["image_grid_thw"] = torch.cat(list(thw_list), dim=0)

        vid_feats = [(f.get("pixel_values_videos", None), f.get("video_grid_thw", None)) for f in features]
        vid_feats = [(pv, thw) for pv, thw in vid_feats if pv is not None]
        if vid_feats:
            pixel_values_list, thw_list = zip(*vid_feats)
            if any(t is None for t in thw_list):
                raise ValueError("`video_grid_thw` is required when `pixel_values_videos` is provided.")
            batch["pixel_values_videos"] = torch.cat(list(pixel_values_list), dim=0)
            batch["video_grid_thw"] = torch.cat(list(thw_list), dim=0)

        return batch


def read_annotations(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if p.suffix == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            return [json.loads(l) for l in f]
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


class PointCloudTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels", None)
        if labels is None:
            raise ValueError("Missing `labels` in inputs.")

        # Call the inner Qwen3VLModel (point-aware) directly to avoid relying on the outer
        # `Qwen3VLForConditionalGeneration.forward` signature for point-cloud-specific keys.
        causal_lm = _unwrap_to_qwen_causal_lm(model)
        core = _get_qwen_core(model)

        core_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        core_outputs = core(**core_inputs)

        hidden_states = getattr(core_outputs, "last_hidden_state", None)
        if hidden_states is None:
            raise ValueError("Inner model did not return `last_hidden_state`.")

        lm_head = getattr(causal_lm, "lm_head", None)
        if lm_head is None:
            raise ValueError("Could not locate `lm_head` on the causal LM wrapper.")

        logits = lm_head(hidden_states)

        # Align logits with text labels by skipping the point-prefix tokens in the sequence.
        pc_seq_len = 0
        pc_embeds = core_inputs.get("point_cloud_embeds", None)
        if isinstance(pc_embeds, torch.Tensor) and pc_embeds.ndim >= 2:
            pc_seq_len = int(pc_embeds.shape[1])
        else:
            pc_adapter_model = getattr(core, "pc_adapter_model", None)
            cfg = getattr(pc_adapter_model, "point_backbone_config", None)
            if isinstance(cfg, dict) and cfg.get("point_token_len", None) is not None:
                pc_seq_len = int(cfg["point_token_len"])

        batch_size, logits_len, vocab_size = logits.shape
        label_len = labels.shape[1]
        expected_len = pc_seq_len + label_len
        if logits_len != expected_len:
            raise ValueError(
                f"Unexpected logits length: got {logits_len}, expected {expected_len} "
                f"(pc_seq_len={pc_seq_len}, label_len={label_len}). "
                "This usually indicates the point prefix was not inserted into the sequence."
            )

        input_ids = core_inputs.get("input_ids", None)
        if isinstance(input_ids, torch.Tensor) and labels.shape != input_ids.shape:
            raise ValueError(f"labels shape {tuple(labels.shape)} != input_ids shape {tuple(input_ids.shape)}")

        if label_len < 2:
            loss = logits.new_zeros(())
        else:
            # Match the standard HF causal LM loss exactly on the augmented (P+L) sequence by
            # prepending -100 labels for the point prefix and applying the usual shift.
            if pc_seq_len > 0:
                ignore = labels.new_full((batch_size, pc_seq_len), -100)
                labels_aug = torch.cat([ignore, labels], dim=1)
            else:
                labels_aug = labels
            shift_labels = labels_aug[:, 1:].contiguous()
            shift_logits = logits[:, :-1, :].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1), ignore_index=-100)

        outputs = {"loss": loss, "logits": logits, "past_key_values": getattr(core_outputs, "past_key_values", None)}
        return (loss, outputs) if return_outputs else loss


class SavePointProjectorCallback(TrainerCallback):
    def __init__(self, interval: Optional[int]):
        self.interval = int(interval) if interval else 0

    def on_step_end(self, args, state, control, **kwargs):
        if self.interval <= 0:
            return
        step = int(getattr(state, "global_step", 0))
        if step <= 0 or (step % self.interval) != 0:
            return
        if not _is_rank0():
            return

        model = kwargs.get("model", None)
        if model is None:
            return
        core = _get_qwen_core(model)
        pc = getattr(core, "pc_adapter_model", None)
        if pc is None or not bool(getattr(core, "pc_train_projector", False)):
            return

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sd: Dict[str, torch.Tensor] = {f"point_proj.{k}": v.detach().cpu() for k, v in pc.point_proj.state_dict().items()}
        if hasattr(pc, "align_mlp") and isinstance(pc.align_mlp, torch.nn.Linear):
            sd.update({f"align_mlp.{k}": v.detach().cpu() for k, v in pc.align_mlp.state_dict().items()})
        out_path = out_dir / f"point_proj_step-{step}.bin"
        torch.save({"state_dict": sd}, str(out_path))


def main():
    parser = argparse.ArgumentParser(description="Train Qwen3-VL with Point Clouds")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        choices=["flash_attention_2", "sdpa", "eager"],
        help="默认自动选择（优先 flash_attention_2）。",
    )
    parser.add_argument("--train-file", type=str, required=True, help="训练标注文件（json/jsonl）")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="<=0 禁用裁剪")
    parser.add_argument("--grad-ckpt", action="store_true")

    parser.add_argument("--save-every-steps", type=int, default=None)
    parser.add_argument("--force-save-checkpoints", action="store_true")
    parser.add_argument("--save-total-limit", type=int, default=2)

    parser.add_argument("--train-visual", action="store_true")
    parser.add_argument("--train-connector", action="store_true")
    parser.add_argument("--train-llm", action="store_true")

    parser.add_argument("--point-backbone", type=str, default="PointBERT")
    parser.add_argument("--point-backbone-config", type=str, default="PointTransformer_8192point_2layer")
    parser.add_argument("--point-use-color", action="store_true")
    parser.add_argument("--point-num", type=int, default=0)
    parser.add_argument("--pc-embeds-device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--point-backbone-ckpt", type=str, default=None)
    parser.add_argument("--point-proj-ckpt", type=str, default=None)
    parser.add_argument("--save-point-proj-ckpt", type=str, default=None)
    parser.add_argument("--save-point-proj-interval", type=int, default=None)
    parser.add_argument("--save-point-backbone-ckpt", type=str, default=None)
    parser.add_argument("--train-point-backbone", action="store_true")
    parser.add_argument("--train-point-projector", action="store_true")

    parser.add_argument("--lora-enable", action="store_true")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-adapter-path", type=str, default=None)
    parser.add_argument("--merge-lora-on-save", action="store_true")

    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)

    args = parser.parse_args()

    if args.lora_enable and not args.train_llm:
        raise ValueError("`--lora-enable` requires `--train-llm`.")

    if args.deepspeed:
        try:
            import deepspeed  # noqa: F401
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("DeepSpeed config is provided but `deepspeed` is not installed.") from e

    attn_impl = args.attn_implementation
    if attn_impl is None:
        attn_impl = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
    elif attn_impl == "flash_attention_2" and not is_flash_attn_2_available():
        attn_impl = "sdpa"

    os.makedirs(args.output_dir, exist_ok=True)

    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        attn_implementation=attn_impl,
        dtype=(torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)),
    )
    qwen.config.use_cache = False

    processor = AutoProcessor.from_pretrained(args.model, fix_mistral_regex=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=False, fix_mistral_regex=True, padding_side="right"
    )

    # Align special tokens across tokenizer/processor/model.
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else tokenizer
    for k in ("pad_token_id", "eos_token_id", "bos_token_id"):
        v = getattr(tok, k, None)
        if v is not None:
            setattr(qwen.config, k, v)
            if hasattr(qwen, "generation_config") and getattr(qwen.generation_config, k, None) is not None:
                setattr(qwen.generation_config, k, v)

    # Register optimizer patch (Trainer.create_optimizer) used by Qwen-VL finetune scripts.
    import qwenvl.train.trainer  # noqa: F401

    # Make the inner model point-cloud aware without copying weights (avoid doubling CPU/GPU memory).
    import types as _types
    qwen.model.forward = _types.MethodType(Qwen3VLModel_PointCloud.forward, qwen.model)

    if args.grad_ckpt:
        if hasattr(qwen, "gradient_checkpointing_enable"):
            qwen.gradient_checkpointing_enable()
        if hasattr(qwen, "enable_input_require_grads"):
            qwen.enable_input_require_grads()
        else:
            def make_inputs_require_grad(_module, _input, output):
                output.requires_grad_(True)
            qwen.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Freeze everything by default; selectively unfreeze below.
    for p in qwen.parameters():
        p.requires_grad = False

    if args.train_visual:
        for p in qwen.model.visual.parameters():
            p.requires_grad = True
    if args.train_connector:
        for p in qwen.model.visual.merger.parameters():
            p.requires_grad = True

    if args.train_llm and not args.lora_enable:
        for p in qwen.model.language_model.parameters():
            p.requires_grad = True
        for p in qwen.lm_head.parameters():
            p.requires_grad = True

    if args.train_llm and args.lora_enable:
        try:
            from peft import LoraConfig, PeftModel, TaskType, get_peft_model  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("LoRA training requires `peft` (pip install peft).") from e

        lora_cfg = None
        if args.lora_adapter_path:
            qwen = PeftModel.from_pretrained(qwen, args.lora_adapter_path)
        else:
            lora_cfg = LoraConfig(
                r=int(args.lora_r),
                lora_alpha=int(args.lora_alpha),
                lora_dropout=float(args.lora_dropout),
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            qwen = get_peft_model(qwen, lora_cfg)

        # Keep lm_head frozen in LoRA mode.
        base = _unwrap_to_qwen_causal_lm(qwen)
        for p in base.lm_head.parameters():
            p.requires_grad = False

    # Build / attach point adapter
    hidden_size = int(_unwrap_to_qwen_causal_lm(qwen).config.text_config.hidden_size)
    train_point_any = bool(args.train_point_backbone or args.train_point_projector)
    if train_point_any:
        pc_adapter_fn, pc_meta, pc_adapter_model = create_pointcloud_adapter(
            hidden_size=hidden_size,
            point_backbone=args.point_backbone,
            point_backbone_config_name=args.point_backbone_config,
            use_color=args.point_use_color,
            point_backbone_ckpt=args.point_backbone_ckpt,
            point_proj_ckpt=args.point_proj_ckpt,
            device=torch.device("cpu"),
            eval_mode=False,
            return_model=True,
        )
    else:
        pc_adapter_fn, pc_meta, pc_adapter_model = create_pointcloud_adapter(
            hidden_size=hidden_size,
            point_backbone=args.point_backbone,
            point_backbone_config_name=args.point_backbone_config,
            use_color=args.point_use_color,
            point_backbone_ckpt=args.point_backbone_ckpt,
            point_proj_ckpt=args.point_proj_ckpt,
            device=torch.device("cpu"),
            eval_mode=True,
            return_model=True,
        )

    if args.point_proj_ckpt:
        report = getattr(pc_adapter_model, "point_proj_load_report", None)
        if report is None:
            raise RuntimeError(f"Failed to load point_proj checkpoint: {args.point_proj_ckpt}")
        missing = list(report.get("missing_keys", []))
        if missing:
            raise RuntimeError(f"point_proj checkpoint missing keys: {missing}")

    if train_point_any:
        core = _get_qwen_core(qwen)
        core.pc_adapter_model = pc_adapter_model
        core.pc_train_backbone = bool(args.train_point_backbone)
        core.pc_train_projector = bool(args.train_point_projector)

        for p in pc_adapter_model.point_backbone.parameters():
            p.requires_grad = bool(args.train_point_backbone)
        for p in pc_adapter_model.point_proj.parameters():
            p.requires_grad = bool(args.train_point_projector)
        if hasattr(pc_adapter_model, "align_mlp"):
            for p in pc_adapter_model.align_mlp.parameters():
                p.requires_grad = bool(args.train_point_projector)

        pc_adapter_model.point_backbone.train() if args.train_point_backbone else pc_adapter_model.point_backbone.eval()
        pc_adapter_model.point_proj.train() if args.train_point_projector else pc_adapter_model.point_proj.eval()
        if hasattr(pc_adapter_model, "align_mlp"):
            pc_adapter_model.align_mlp.train() if args.train_point_projector else pc_adapter_model.align_mlp.eval()

    if not any(p.requires_grad for p in qwen.parameters()):
        raise ValueError(
            "No trainable parameters detected. Enable at least one of "
            "`--train-visual/--train-connector/--train-llm/--train-point-*`."
        )

    anns = read_annotations(args.train_file)
    embeds_out_device = torch.device(
        "cuda" if (args.pc_embeds_device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    dataset = PointCloudVisualDataset(
        annotations=anns,
        processor=processor,
        pc_adapter_fn=pc_adapter_fn,
        pc_token_len=int(pc_meta["point_token_len"]),
        hidden_size=hidden_size,
        use_color=args.point_use_color,
        point_num=args.point_num,
        embeds_out_device=embeds_out_device,
    )
    dataset.use_pc_adapter_forward = bool(train_point_any)

    collator = PointCloudCollator(tokenizer)
    collator.use_pc_adapter_forward = bool(train_point_any)

    train_any_qwen_component = bool(args.train_visual or args.train_connector or args.train_llm)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=bool(args.grad_ckpt),
        gradient_accumulation_steps=max(1, int(args.grad_accum_steps)),
        max_grad_norm=(0.0 if float(args.max_grad_norm) <= 0 else float(args.max_grad_norm)),
        ddp_find_unused_parameters=True,
        logging_steps=10,
        save_steps=(args.save_every_steps if args.save_every_steps is not None else 1000),
        save_total_limit=(None if args.save_total_limit is not None and int(args.save_total_limit) <= 0 else args.save_total_limit),
        save_strategy=("steps" if (train_any_qwen_component or args.force_save_checkpoints) else "no"),
        remove_unused_columns=False,
        lr_scheduler_type=str(args.lr_scheduler),
        warmup_steps=int(args.warmup_steps),
        warmup_ratio=float(args.warmup_ratio),
        report_to="tensorboard",
        logging_dir=os.path.join(args.output_dir, "runs"),
        deepspeed=(str(args.deepspeed) if args.deepspeed else None),
    )

    trainer = PointCloudTrainer(
        model=qwen,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    if args.save_point_proj_interval and train_point_any and args.train_point_projector:
        trainer.add_callback(SavePointProjectorCallback(args.save_point_proj_interval))

    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    if train_any_qwen_component or args.force_save_checkpoints:
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)

    if args.lora_enable and args.merge_lora_on_save and _is_rank0():
        merged_dir = os.path.join(args.output_dir, "merged")
        os.makedirs(merged_dir, exist_ok=True)
        merged_model = qwen.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        processor.save_pretrained(merged_dir)

    if _is_rank0():
        core = _get_qwen_core(qwen)
        pc = getattr(core, "pc_adapter_model", None)
        if pc is not None and args.save_point_proj_ckpt and args.train_point_projector:
            sd: Dict[str, torch.Tensor] = {f"point_proj.{k}": v.detach().cpu() for k, v in pc.point_proj.state_dict().items()}
            if hasattr(pc, "align_mlp") and isinstance(pc.align_mlp, torch.nn.Linear):
                sd.update({f"align_mlp.{k}": v.detach().cpu() for k, v in pc.align_mlp.state_dict().items()})
            torch.save({"state_dict": sd}, args.save_point_proj_ckpt)
        if pc is not None and args.save_point_backbone_ckpt and args.train_point_backbone:
            bk_sd = {f"module.point_encoder.{k}": v.detach().cpu() for k, v in pc.point_backbone.state_dict().items()}
            torch.save({"state_dict": bk_sd}, args.save_point_backbone_ckpt)


if __name__ == "__main__":
    main()

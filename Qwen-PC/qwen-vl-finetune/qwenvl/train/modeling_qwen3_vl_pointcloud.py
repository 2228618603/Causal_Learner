import torch
from typing import Optional, Union, Any

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel as HFQwen3VLModel,
    Qwen3VLModelOutputWithPast,
)
from transformers.utils import is_torchdynamo_compiling

class Qwen3VLModel_PointCloud(HFQwen3VLModel):
    """
    基于官方 Qwen3VLModel 的点云扩展版本：仅覆写 forward，保持权重完全兼容。

    要点：
    - 新增参数 point_cloud_embeds: (batch, pc_seq_len, hidden_size)
    - 将点云嵌入前置拼接到 inputs_embeds 开头
    - attention_mask 前置 1；mRoPE 位置整体右移并拼接；DeepStack 不作用于点云
    - 其余行为与官方 Qwen3VLModel 保持一致
    """

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[Union[torch.Tensor, dict]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        point_cloud_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        # 仅允许 input_ids 或 inputs_embeds 其一
        # 应为“恰好指定其中一个”，因此当两者同为空或同非空时抛错
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # 可选：点云适配器模块（由外部脚本注入），用于在前向中端到端生成嵌入
        pc_adapter_model = getattr(self, "pc_adapter_model", None)
        # 细粒度控制：分别控制 PointBERT 与 projector 的可训练性
        pc_train_backbone = bool(getattr(self, "pc_train_backbone", False))
        pc_train_projector = bool(getattr(self, "pc_train_projector", False))

        # 从 kwargs 中提取原始点云（可为 list[Tensor] 或 Tensor），以支持前向计算嵌入
        point_clouds = kwargs.pop("point_clouds", None)
        if point_cloud_embeds is None and point_clouds is not None and pc_adapter_model is None:
            raise ValueError(
                "`point_clouds` was provided but no `pc_adapter_model` is attached. "
                "Attach an adapter (core.pc_adapter_model=...) or pass `point_cloud_embeds` directly."
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        def _rope_attention_mask(mask: Optional[Union[torch.Tensor, dict]]) -> Optional[torch.Tensor]:
            # get_rope_index expects a 2D (B, L) mask where 1 means "not masked".
            if mask is None:
                return None

            mask_tensor: Optional[torch.Tensor]
            if isinstance(mask, dict):
                mask_tensor = mask.get("attention_mask", None)
                if mask_tensor is None:
                    mask_tensor = mask.get("full_attention", None)
            else:
                mask_tensor = mask

            if not isinstance(mask_tensor, torch.Tensor):
                return None

            if mask_tensor.ndim == 4:
                mask_tensor = torch.diagonal(mask_tensor[:, 0], dim1=1, dim2=2)

            if mask_tensor.ndim != 2:
                return None

            if mask_tensor.dtype == torch.bool:
                return mask_tensor.to(dtype=torch.int32)

            if mask_tensor.dtype.is_floating_point:
                finfo = torch.finfo(mask_tensor.dtype)
                masked = (mask_tensor == finfo.min) | (mask_tensor <= -1e30) | (torch.isinf(mask_tensor) & (mask_tensor < 0))
                return (~masked).to(dtype=torch.int32)

            return mask_tensor

        # 图像/视频特征注入（保持官方逻辑）
        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # DeepStack 视觉特征（点云不参与）
        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        # 计算 position_ids / rope_deltas（保持官方逻辑）
        if not hasattr(self, "rope_deltas"):
            self.rope_deltas = None

        if position_ids is None:
            attention_mask_tensor = _rope_attention_mask(attention_mask)

            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1) or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = (not is_torchdynamo_compiling()) and (
                (cache_position is not None and cache_position[0] == 0) or (past_key_values is None or past_key_values.get_seq_length() == 0 if hasattr(past_key_values, 'get_seq_length') else True)
            )

            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device).view(1, -1).expand(batch_size, -1)
                if cache_position is not None and isinstance(delta, torch.Tensor):
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # 若未提供外部点云嵌入，但提供了原始点云且存在适配器模块，则在前向中生成嵌入
        if point_cloud_embeds is None and point_clouds is not None and pc_adapter_model is not None:
            # 统一到适配器设备
            pc_dev = next(pc_adapter_model.parameters()).device
            # 计算每样本嵌入；支持 list[Tensor] 或 (B, N, D) Tensor
            pcs: list = []
            if isinstance(point_clouds, torch.Tensor):
                # (B, N, D) → list[Tensor{N,D}]
                for i in range(point_clouds.shape[0]):
                    pcs.append(point_clouds[i])
            elif isinstance(point_clouds, list):
                pcs = point_clouds
            else:
                pcs = []

            embeds_list = []
            # 适配器配置与输出形状
            cfg = pc_adapter_model.point_backbone_config
            p_tokens = cfg.get("point_token_len", 1)
            hidden = self.config.text_config.hidden_size

            def _first_out(x: Any) -> torch.Tensor:
                return x[0] if isinstance(x, (tuple, list)) else x

            for pc in pcs:
                if pc is None:
                    embeds_list.append(torch.zeros(p_tokens, hidden, device=pc_dev))
                    continue
                pc_in = pc.to(device=pc_dev)
                # Backbone 前向：根据 pc_train_backbone 决定是否记录梯度
                if pc_train_backbone:
                    feats = _first_out(pc_adapter_model.point_backbone(pc_in.unsqueeze(0)))
                else:
                    with torch.no_grad():
                        feats = _first_out(pc_adapter_model.point_backbone(pc_in.unsqueeze(0)))
                # Projector + 对齐层：
                # - 若仅训练 backbone，需要梯度能回传到 backbone，因此 projector 前向仍需保留计算图；
                # - 若训练 projector，本身也需要梯度；
                # - 若两者均不训练，则无梯度。
                projector_grad_enabled = (pc_train_backbone or pc_train_projector)
                if projector_grad_enabled:
                    emb = _first_out(pc_adapter_model.point_proj(feats))
                    emb = _first_out(pc_adapter_model.align_mlp(emb))
                else:
                    with torch.no_grad():
                        emb = _first_out(pc_adapter_model.point_proj(feats))
                        emb = _first_out(pc_adapter_model.align_mlp(emb))
                embeds_list.append(emb.squeeze(0))

            if len(embeds_list) > 0:
                point_cloud_embeds = torch.stack(embeds_list, dim=0)

        # 点云注入 + mRoPE 位置处理
        if point_cloud_embeds is not None:
            # Preserve the original (pre-prefix) padding mask for position shifting.
            # We intentionally shift only non-padding tokens by +P so padding positions remain fixed (usually 1),
            # matching Qwen3-VL/rope2d conventions.
            rope_shift_mask = _rope_attention_mask(attention_mask)
            # 维度断言：确保点云嵌入的隐藏维与文本 hidden_size 一致
            expected_hidden = int(self.config.text_config.hidden_size)
            if point_cloud_embeds.ndim != 3:
                raise ValueError(
                    f"point_cloud_embeds must be a 3D tensor of shape (batch, pc_seq_len, hidden_size), got {point_cloud_embeds.shape}"
                )
            if point_cloud_embeds.shape[-1] != expected_hidden:
                raise ValueError(
                    f"point_cloud_embeds hidden dim {point_cloud_embeds.shape[-1]} != model hidden_size {expected_hidden}"
                )
            # Ensure we have base position_ids for the original (pre-prefix) sequence.
            # Qwen3-VL uses mRoPE (3, B, L); if callers do not provide it we must compute it from input_ids.
            if position_ids is None:
                if input_ids is None:
                    raise ValueError(
                        "position_ids is required when using point_cloud_embeds with inputs_embeds-only inputs"
                    )
                attention_mask_tensor = _rope_attention_mask(attention_mask)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            orig_seq_len = int(inputs_embeds.shape[1])
            pc = point_cloud_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            batch_size, pc_seq_len, _ = pc.shape
            if batch_size != inputs_embeds.shape[0]:
                raise ValueError(
                    f"Batch size mismatch: input batch={int(inputs_embeds.shape[0])}, point_cloud_embeds batch={batch_size}"
                )
            if position_ids.ndim != 3 or position_ids.shape[0] != 3 or position_ids.shape[1] != batch_size:
                raise ValueError(
                    f"position_ids must have shape (3, batch, seq_len) for Qwen3-VL mRoPE, got {tuple(position_ids.shape)}"
                )
            if int(position_ids.shape[2]) != orig_seq_len:
                raise ValueError(
                    f"position_ids seq_len mismatch: got {int(position_ids.shape[2])}, expected {orig_seq_len}"
                )
            inputs_embeds = torch.cat([pc, inputs_embeds], dim=1)

            if attention_mask is not None:
                if isinstance(attention_mask, dict):
                    # If a simple 2D padding mask is provided in the dict, keep it in sync with the
                    # augmented (P+L) sequence length by prepending visible point tokens.
                    base_am = attention_mask.get("attention_mask", None)
                    if isinstance(base_am, torch.Tensor) and base_am.ndim == 2:
                        if base_am.shape[0] != batch_size or int(base_am.shape[1]) != orig_seq_len:
                            raise ValueError(
                                f"attention_mask['attention_mask'] shape {tuple(base_am.shape)} does not match "
                                f"batch/seq_len {(batch_size, orig_seq_len)}"
                            )
                        if base_am.dtype.is_floating_point:
                            pc_mask = base_am.new_zeros((batch_size, pc_seq_len))
                        else:
                            pc_mask = torch.ones((batch_size, pc_seq_len), dtype=base_am.dtype, device=base_am.device)
                        attention_mask["attention_mask"] = torch.cat([pc_mask, base_am], dim=1)

                    fa = attention_mask.get("full_attention", None)
                    if fa is not None:
                        # 保持 full_attention 的形态：
                        # - 若为 4D (B, H, L, L)，则扩展为 (B, H, P+L, P+L) 并将原块置于右下角，
                        #   点云前缀与交叉块默认允许可见（置 0）。
                        # - 若为 2D (B, L)，则前拼点云可见位（1），保持 2D 语义。
                        if fa.ndim == 4:
                            B, H, L1, L2 = fa.shape
                            # 兼容非常规情况下的非方阵输入
                            L = L1 if L1 == L2 else min(L1, L2)
                            # full_attention 通常为加性 mask（0=可见，-inf/min=遮蔽），需保持因果性：
                            # - 点云前缀之间为严格因果（上三角遮蔽）
                            # - 点云前缀不能看见后续文本/视觉 token（整块遮蔽）
                            # - 文本/视觉 token 可以看见点云前缀（若该 query 非 padding）
                            if fa.dtype.is_floating_point:
                                masked_value = torch.finfo(fa.dtype).min
                                fa_new = fa.new_full((B, H, pc_seq_len + L, pc_seq_len + L), masked_value)
                                # 原序列 mask 放到右下角
                                fa_new[:, :, pc_seq_len:, pc_seq_len:] = fa[:, :, :L, :L]
                                # 文本/视觉 query -> 点云 key：默认可见（0），padding query 行保持遮蔽
                                fa_new[:, :, pc_seq_len:, :pc_seq_len] = 0
                                diag = torch.diagonal(fa[:, :, :L, :L], dim1=-2, dim2=-1)  # (B, H, L)
                                pad_q = (diag == masked_value) | (diag <= -1e30) | (torch.isinf(diag) & (diag < 0))
                                if pad_q.any():
                                    fa_new[:, :, pc_seq_len:, :pc_seq_len] = fa_new[:, :, pc_seq_len:, :pc_seq_len].masked_fill(
                                        pad_q.unsqueeze(-1), masked_value
                                    )
                                # 点云前缀内部因果 mask
                                tri = torch.triu(
                                    torch.ones((pc_seq_len, pc_seq_len), dtype=torch.bool, device=fa.device), diagonal=1
                                )
                                fa_new[:, :, :pc_seq_len, :pc_seq_len] = 0
                                fa_new[:, :, :pc_seq_len, :pc_seq_len] = fa_new[:, :, :pc_seq_len, :pc_seq_len].masked_fill(
                                    tri, masked_value
                                )
                            else:
                                # 保守兜底：保持原 mask 右下角，新增区域默认可见
                                fa_new = fa.new_zeros((B, H, pc_seq_len + L, pc_seq_len + L))
                                fa_new[:, :, pc_seq_len:, pc_seq_len:] = fa[:, :, :L, :L]
                            attention_mask["full_attention"] = fa_new
                        else:
                            # 2D 语义下按可见位前拼
                            if fa.dtype == torch.bool:
                                vis2d = fa.to(torch.int)
                            elif fa.dtype.is_floating_point:
                                finfo = torch.finfo(fa.dtype)
                                masked = (fa == finfo.min) | (fa <= -1e30)
                                vis2d = (~masked).to(torch.int)
                            else:
                                vis2d = fa.to(torch.int)
                            pc_mask = torch.ones((batch_size, pc_seq_len), dtype=vis2d.dtype, device=vis2d.device)
                            attention_mask["full_attention"] = torch.cat([pc_mask, vis2d], dim=1)
                    else:
                        # 兼容 varlen：如提供了 cu_seqlens_q/cu_seqlens，则将每样本长度加上点云前缀 P
                        cu = attention_mask.get("cu_seqlens_q", None)
                        if cu is None:
                            cu = attention_mask.get("cu_seqlens", None)
                        if cu is not None and isinstance(cu, torch.Tensor) and cu.ndim == 1:
                            cu = cu.to(dtype=torch.int32, device=inputs_embeds.device)
                            lens = (cu[1:] - cu[:-1]).to(torch.int32)
                            # 为每个样本长度加上点云 token 数
                            lens = lens + torch.tensor([pc_seq_len] * int(lens.numel()), dtype=torch.int32, device=lens.device)
                            new_cu = torch.zeros((lens.numel() + 1,), dtype=torch.int32, device=lens.device)
                            new_cu[1:] = torch.cumsum(lens, dim=0)
                            if "cu_seqlens_q" in attention_mask:
                                attention_mask["cu_seqlens_q"] = new_cu
                            if "cu_seqlens_k" in attention_mask:
                                attention_mask["cu_seqlens_k"] = new_cu
                            if "cu_seqlens" in attention_mask:
                                attention_mask["cu_seqlens"] = new_cu
                            for k in ("max_seqlen_q", "max_seqlen_k", "max_seqlen"):
                                if k in attention_mask:
                                    v = attention_mask[k]
                                    if isinstance(v, torch.Tensor):
                                        attention_mask[k] = v + pc_seq_len
                                    elif isinstance(v, int):
                                        attention_mask[k] = int(v) + pc_seq_len
                        # 若未提供支持的键，保持原样
                elif isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
                    pc_mask = torch.ones((batch_size, pc_seq_len), dtype=attention_mask.dtype, device=attention_mask.device)
                    attention_mask = torch.cat([pc_mask, attention_mask], dim=1)

            # Cache positions are indexed over the *actual* sequence length fed into the language model.
            # When we prepend point tokens during the prefill step, we must extend `cache_position` accordingly
            # so KV cache bookkeeping stays consistent under `use_cache=True` / generation.
            if cache_position is not None and isinstance(cache_position, torch.Tensor) and cache_position.ndim == 1:
                if cache_position.numel() == orig_seq_len and cache_position.numel() > 0 and cache_position[0].item() == 0:
                    cache_position = torch.arange(
                        pc_seq_len + orig_seq_len, device=cache_position.device, dtype=cache_position.dtype
                    )

            # Prepending a point prefix of length P: shift *all* existing position ids by +P (text and vision)
            # so that relative positions among original tokens stay unchanged, while point tokens occupy [0..P-1].
            if rope_shift_mask is not None and isinstance(rope_shift_mask, torch.Tensor) and rope_shift_mask.ndim == 2:
                if rope_shift_mask.shape[0] != batch_size or rope_shift_mask.shape[1] != orig_seq_len:
                    raise ValueError(
                        f"attention_mask shape {tuple(rope_shift_mask.shape)} does not match "
                        f"batch/seq_len {(batch_size, orig_seq_len)}"
                    )
                shift = (rope_shift_mask.to(dtype=position_ids.dtype, device=position_ids.device) * pc_seq_len).unsqueeze(0)
                pos = position_ids + shift.expand(3, -1, -1)
            else:
                pos = position_ids + pc_seq_len
            t = torch.arange(pc_seq_len, device=position_ids.device).view(1, -1).expand(batch_size, -1)
            pc_pos = position_ids.new_zeros((3, batch_size, pc_seq_len))
            pc_pos[0] = t
            pc_pos[1] = t
            pc_pos[2] = t
            position_ids = torch.cat([pc_pos, pos], dim=-1)

            if visual_pos_masks is not None:
                pc_vis = torch.zeros((batch_size, pc_seq_len), dtype=visual_pos_masks.dtype, device=visual_pos_masks.device)
                visual_pos_masks = torch.cat([pc_vis, visual_pos_masks], dim=1)

        # 进入文本模型
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )

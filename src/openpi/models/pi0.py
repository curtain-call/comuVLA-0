import dataclasses
import logging
from typing import Any

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array
from jaxlib.xla_extension import batched_block_until_ready
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
        pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_600m"

    # Set the model specific defaults.
    action_dim: int = 7
    action_horizon: int = 50
    max_token_len: int = 64

    # Atomic supervision
    atomic_vocab_sizes: tuple[int, int, int, int] = (64, 32, 3, 8)
    loss_weights: dict[str, float] = dataclasses.field(default_factory=lambda: {"action": 1.0, "atomic": 1.0, "ntp": 0.0})
    # New options: condition mode and kv policy
    cond_mode: str = "state"  # one of: "state", "atomic", "fuse"
    kv_policy: str = "placeholder"   # one of: "full", "placeholder"

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


class Pi0(_model.BaseModel):
    def __init__(self, config: Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        # Cache paligemma width for later use
        self.paligemma_width = paligemma_config.width
        # Projectors / heads
        self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        # Atomic token embedding和条件融合仅在需要时构建（cond_mode in {"atomic","fuse"}）
        if config.cond_mode in ("atomic", "fuse"):
            self.atomic_embed_t = nnx.Embed(config.atomic_vocab_sizes[0], action_expert_config.width // 4, rngs=rngs)
            self.atomic_embed_r = nnx.Embed(config.atomic_vocab_sizes[1], action_expert_config.width // 4, rngs=rngs)
            self.atomic_embed_g = nnx.Embed(config.atomic_vocab_sizes[2], action_expert_config.width // 8, rngs=rngs)
            self.atomic_embed_d = nnx.Embed(config.atomic_vocab_sizes[3], action_expert_config.width // 8, rngs=rngs)
            self.atomic_mlp = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.cond_fuse = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.atomic_embed_t = None
            self.atomic_embed_r = None
            self.atomic_embed_g = None
            self.atomic_embed_d = None
            self.atomic_mlp = None
            self.cond_fuse = None
        # Classification heads from pooled prefix output
        self.atomic_head_t = nnx.Linear(self.paligemma_width, config.atomic_vocab_sizes[0], rngs=rngs)
        self.atomic_head_r = nnx.Linear(self.paligemma_width, config.atomic_vocab_sizes[1], rngs=rngs)
        self.atomic_head_g = nnx.Linear(self.paligemma_width, config.atomic_vocab_sizes[2], rngs=rngs)
        self.atomic_head_d = nnx.Linear(self.paligemma_width, config.atomic_vocab_sizes[3], rngs=rngs)
        # placeholder-only 模式下，我们不从原子 token 生成占位前缀，避免将原子信息显式注入后缀；
        # 后缀所需语义由 state 或 cond_mode 融合承担，kv_cache 仅提供结构性前缀位点。
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)


    @at.typecheck
    def embed_prefix(
            self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]
            # ar_mask.append(jnp.zeros((image_tokens.shape[1]), dtype=bool))

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs

            # 原始实现：文本token之间互相可见，未建模语序
            ar_mask += [False] * tokenized_inputs.shape[1]
            # 修改为自回归建模
            # bs, ss = obs.tokenized_prompt.shape
            # q_len = obs.question_len[0]
            # # ar_mask += [False] * q_len + [True] * (ss - q_len)
            # ar_mask = jnp.arange(len(ar_mask) + ss) >= (jnp.asarray(len(ar_mask)) + q_len)
            # print(ar_mask)

            # big_vision实现
            # ar_mask += [True] * tokenized_inputs.shape[1]


        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
            self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # condition token: always include state; optionally add atomic tokens per cond_mode
        cond_token: at.Array
        state_cond = self.state_proj(obs.state)
        cond_mode = getattr(self.config, "cond_mode", "fuse")
        if cond_mode in ("atomic", "fuse") and (self.atomic_embed_t is not None) and obs.atomic_tokens is not None:
            t_idx, r_idx, g_idx, d_idx = (
                obs.atomic_tokens[..., 0],
                obs.atomic_tokens[..., 1],
                obs.atomic_tokens[..., 2],
                obs.atomic_tokens[..., 3],
            )
            t_emb = self.atomic_embed_t(t_idx)
            r_emb = self.atomic_embed_r(r_idx)
            g_emb = self.atomic_embed_g(g_idx)
            d_emb = self.atomic_embed_d(d_idx)
            atomic_cond = jnp.concatenate([t_emb, r_emb, g_emb, d_emb], axis=-1)
            atomic_cond = self.atomic_mlp(atomic_cond)
            if cond_mode == "atomic":
                # simple additive fusion
                fused = state_cond + atomic_cond
                cond_token = fused[:, None, :]
            else:  # fuse
                fused = jnp.concatenate([state_cond, atomic_cond], axis=-1)
                fused = self.cond_fuse(fused)
                cond_token = fused[:, None, :]
        else:
            cond_token = state_cond[:, None, :]
        tokens.append(cond_token)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        # image/language inputs do not attend to state or actions
        ar_mask += [True]

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        # mix timestep + action information using an MLP
        action_tokens = self.action_in_proj(noisy_actions)
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)
        tokens.append(action_time_tokens)
        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @override
    def compute_loss(
            self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[Array, float | Any]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time)



        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        # prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)

        num_image_tokens = 256
        num_text_tokens = observation.tokenized_prompt.shape[1]

        # Policies for what the suffix can attend to
        kv_policy = getattr(self.config, "kv_policy", "full")
        if kv_policy == "placeholder":
            # 1) Build placeholder tokens and append after image+text so placeholders can attend to them
            batch = observation.state.shape[0]
            pal_width = self.paligemma_width
            P = 4
            ph_tokens = jnp.zeros((batch, P, pal_width), dtype=jnp.float32)
            ph_mask = jnp.ones((batch, P), dtype=jnp.bool_)
            # start a new block at first placeholder, rest in same block
            ph_ar = jnp.array([True] + [False] * (P - 1))

            prefix_ext = jnp.concatenate([prefix_tokens, ph_tokens], axis=1)
            prefix_ext_mask = jnp.concatenate([prefix_mask, ph_mask], axis=1)
            prefix_ext_ar = jnp.concatenate([prefix_ar_mask, ph_ar], axis=0)
            prefix_ext_attn = make_attn_mask(prefix_ext_mask, prefix_ext_ar)
            prefix_ext_pos = jnp.cumsum(prefix_ext_mask, axis=1) - 1

            # Run prefix to get placeholder outputs and full kv_cache (observation + placeholders)
            (prefix_ext_out, _), kv_cache = self.PaliGemma.llm([prefix_ext, None], mask=prefix_ext_attn, positions=prefix_ext_pos)
            placeholder_out = prefix_ext_out[:, -P:, :]

            # 仅保留 视觉(image) + 占位符(placeholder) 的 KV，剔除文本部分
            cache_k, cache_v = kv_cache
            # 计算 I/T/P：图像 token 数、文本 token 数、占位符数
            T = int(observation.tokenized_prompt.shape[1]) if observation.tokenized_prompt is not None else 0
            total_prefix_ext_len = int(prefix_ext_mask.shape[1])
            I = total_prefix_ext_len - T - P
            # 选择 [0:I) 与 [I+T:I+T+P)
            k_img = cache_k[:, :, :I, ...]
            k_ph  = cache_k[:, :, I + T:I + T + P, ...]
            v_img = cache_v[:, :, :I, ...]
            v_ph  = cache_v[:, :, I + T:I + T + P, ...]
            kv_cache = (jnp.concatenate([k_img, k_ph], axis=2), jnp.concatenate([v_img, v_ph], axis=2))

            # 视觉+占位符 的前缀可见性掩码
            vision_mask = prefix_ext_mask[:, :I]
            ph_mask_sel = prefix_ext_mask[:, I + T:I + T + P]
            vision_ph_mask = jnp.concatenate([vision_mask, ph_mask_sel], axis=1)

            # 2) 后缀仅看 视觉+占位符 的 KV
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attend_mask = einops.repeat(vision_ph_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attend_mask, suffix_attn_mask], axis=-1)
            positions_suffix = jnp.sum(vision_ph_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (prefix_out, suffix_out), _ = self.PaliGemma.llm([None, suffix_tokens], mask=full_attn_mask, positions=positions_suffix, kv_cache=kv_cache)
            # use placeholder_out for atomic classification pooling
            prefix_for_cls = placeholder_out
        else:
            # Full prefix available to suffix
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions
            )
            prefix_for_cls = prefix_out


        # text_out = prefix_out[:, num_image_tokens:num_image_tokens+num_text_tokens, :]
        # ntp_loss = 0.0
        # if hasattr(observation, "tokenized_prompt") and observation.tokenized_prompt is not None:
        #     tokens = observation.tokenized_prompt
        #     mask = observation.tokenized_prompt_mask
        #     logits = text_out
        #     input_mask = mask[:, :-1]
        #     target_mask = mask[:, 1:]
        #     ntp_loss = optax.softmax_cross_entropy_with_integer_labels(
        #         logits[:, :-1], tokens[:, :-1]
        #     )
        #     ntp_loss = ntp_loss * target_mask
        #     ntp_loss = ntp_loss.sum() / (target_mask.sum() + 1e-8)

        # 使用suffix_outputs做动作预测
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        # jax.debug.print("v_t shape={s} min={mn} max={mx} mean={me}",
        #                 s=v_t.shape, mn=v_t.min(), mx=v_t.max(), me=v_t.mean())
        # jax.debug.print("u_t shape={s} min={mn} max={mx} mean={me}",
        #                 s=u_t.shape, mn=u_t.min(), mx=u_t.max(), me=u_t.mean())
        diff = v_t - u_t
        # jax.debug.print("diff min={mn} max={mx} mean={me}", mn=diff.min(), mx=diff.max(), me=diff.mean())
        per_step_action_loss = jnp.mean(jnp.square(diff), axis=-1)  # [b, H]
        # jax.debug.print("per_step_action_loss shape={s} min={mn} max={mx} mean={me}",
        #                 s=per_step_action_loss.shape,
        #                 mn=per_step_action_loss.min(), mx=per_step_action_loss.max(), me=per_step_action_loss.mean())
        if observation.atomic_valid is not None:
            valid = observation.atomic_valid.astype(jnp.float32)
            # jax.debug.print("valid sum={vs}", vs=valid.sum())
            action_loss = (per_step_action_loss * valid).sum() / (valid.sum() + 1e-8)
        else:
            action_loss = per_step_action_loss.mean()
        # jax.debug.print("action_loss={al}", al=action_loss)

        # Atomic token classification from pooled prefix outputs
        pooled = jnp.mean(prefix_for_cls, axis=1)
        logits_t = self.atomic_head_t(pooled)
        logits_r = self.atomic_head_r(pooled)
        logits_g = self.atomic_head_g(pooled)
        logits_d = self.atomic_head_d(pooled)
        atomic_loss = 0.0
        if observation.atomic_tokens is not None:
            y = observation.atomic_tokens
            atomic_loss += optax.softmax_cross_entropy_with_integer_labels(logits_t, y[..., 0]).mean()
            atomic_loss += optax.softmax_cross_entropy_with_integer_labels(logits_r, y[..., 1]).mean()
            atomic_loss += optax.softmax_cross_entropy_with_integer_labels(logits_g, y[..., 2]).mean()
            atomic_loss += optax.softmax_cross_entropy_with_integer_labels(logits_d, y[..., 3]).mean()

        w = self.config.loss_weights
        total = w.get("action", 1.0) * action_loss + w.get("atomic", 1.0) * atomic_loss + w.get("ntp", 0.0) * 0

        return total, {"action": action_loss, "atomic": atomic_loss, "ntp": 0}

    @override
    def sample_actions(
            self,
            rng: at.KeyArrayLike,
            observation: _model.Observation,
            *,
            num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> tuple[_model.Actions, list[str]]:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def sample_text(self,
                    rng: at.KeyArrayLike,
                    observation: dict,
                    *,
                    max_gen_len: int=100,
                    tokenizer=None,
                    do_sample: bool = True,
                    temperature: float = 1.0,
                    top_k: int = 50,
                    top_p: float = 0.9
                    ) -> list:
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        # 0) 嵌入表替换
        # 检查嵌入表的数值范围，确认是否为 PaliGemma 权重
        llm_embedder = self.PaliGemma.llm.embedder
        if hasattr(llm_embedder, 'input_embedding_table'):
            embed_table = llm_embedder.input_embedding_table
            print(f"嵌入表形状: {embed_table.shape}")
            print(f"嵌入表数值范围: [{embed_table.min():.4f}, {embed_table.max():.4f}]")
            print(f"嵌入表均值: {embed_table.mean():.4f}")

        # # 1) 词表大小与特殊符号
        # print("tok vocab:", tokenizer.vocab_size)  # 期望 257152
        # print("BOS/EOS:", tokenizer.bos_token, tokenizer.eos_token)
        #
        # # 2) 回环测试：tokenize -> decode
        # s = "caption en: a photo of a cat on the table."
        # ids, mask, _ = tokenizer.tokenize(s)
        # print("roundtrip:", tokenizer.decode(ids.tolist()))
        #
        #
        # TODO 将prefill流程放在单独的方法实现中
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        # 提取最后一个有效位置的 logits
        last_valid_pos = jnp.sum(prefix_mask, axis=1) - 1  # [batch_size]
        batch_indices = jnp.arange(batch_size)
        current_logits = prefix_out[batch_indices, last_valid_pos, :]  # [batch_size, vocab_size]

        # 在 src/openpi/models/pi0.py 的 sample_text 内，替换 sample_token 实现：

        def _mask_logits(logits, tok_mask=None):
            if tok_mask is None:
                return logits
            return jnp.where(tok_mask, jnp.full_like(logits, -1.0e7), logits)

        def _greedy_sampling(logits):
            return jnp.argmax(logits, axis=-1)

        def _temperature_sampling(logits, rng_key, t: float):
            return jax.random.categorical(rng_key, logits / t, axis=-1)

        def _nucleus_sampling(logits, rng_key, p: float, t: float = 1.0):
            logits = logits / t
            neg_inf = jnp.array(-1.0e7, dtype=logits.dtype)
            # 排序并计算累计概率
            logits_sorted = jnp.sort(logits, axis=-1)[:, ::-1]
            probs_sorted = jax.nn.softmax(logits_sorted, axis=-1)
            cumsum = jnp.cumsum(probs_sorted, axis=-1)
            # 保留累计概率 <= p 的最小集合（至少一个）
            keep_count = jnp.sum(cumsum < p, axis=-1, keepdims=True)
            cutoff = jnp.take_along_axis(logits_sorted, keep_count, axis=-1)
            logits = jnp.where(logits < cutoff, neg_inf, logits)
            return jax.random.categorical(rng_key, logits, axis=-1)

        # 你原本的 sample_token 改为：
        def sample_token(logits, rng_key, step, *, eos_id, tokenizer, min_gen_len: int,
                         sampler: str = "greedy", temperature: float = 1.0, nucleus_p: float = 0.9):
            # 1) 组装 tok_mask：首步/最小长度内屏蔽 EOS；屏蔽 PAD/BOS；可选屏蔽 loc/seg
            vsize = logits.shape[-1]
            tok_mask = jnp.zeros_like(logits, dtype=bool)
            try:
                if hasattr(tokenizer, "pad_token"):
                    tok_mask = tok_mask.at[:, tokenizer.pad_token].set(True)
                if hasattr(tokenizer, "bos_token"):
                    tok_mask = tok_mask.at[:, tokenizer.bos_token].set(True)
                if step < min_gen_len:
                    tok_mask = tok_mask.at[:, eos_id].set(True)
            except Exception:
                pass
            # 可选：屏蔽 loc/seg（最后 1152 个）
            try:
                ban_start = tokenizer.vocab_size - (1024 + 128)
                ids = jnp.arange(vsize)[None, :]
                tok_mask = jnp.logical_or(tok_mask, ids >= ban_start)
            except Exception:
                pass

            masked_logits = _mask_logits(logits, tok_mask)
            # 2) 依据 big_vision 语义采样
            if sampler == "greedy":
                return _greedy_sampling(masked_logits)
            elif sampler == "temperature":
                return _temperature_sampling(masked_logits, rng_key, max(temperature, 1e-6))
            elif sampler == "nucleus":
                return _nucleus_sampling(masked_logits, rng_key, p=nucleus_p, t=max(temperature, 1e-6))
            else:
                # 默认回退到 greedy
                return _greedy_sampling(masked_logits)

        # # 2) 采样第一个生成 token
        # def sample_token(logits, rng_key, step):
        #     # 屏蔽 PAD
        #     if hasattr(tokenizer, 'pad_token'):
        #         logits = logits.at[:, tokenizer.pad_token].set(-1e30)
        #     # 屏蔽首步的 EOS，避免空串或异常起始词
        #     try:
        #         if step == 0:
        #             eos_id = tokenizer.eos_token
        #             logits = logits.at[:, eos_id].set(-1e30)
        #     except Exception:
        #         pass
        #
        #     if do_sample and temperature > 0.0:
        #         scaled_logits = logits / temperature
        #         if top_k > 0:
        #             top_logits, top_indices = jax.lax.top_k(scaled_logits, top_k)
        #             probs = jax.nn.softmax(top_logits, axis=-1)
        #             sampled_idx = jax.random.categorical(rng_key, jnp.log(probs), axis=-1)
        #             return jnp.take_along_axis(top_indices, sampled_idx[:, None], axis=-1)[:, 0]
        #         else:
        #             return jax.random.categorical(rng_key, scaled_logits, axis=-1)
        #     else:
        #         return jnp.argmax(logits, axis=-1)

        # 3) 初始化生成状态
        generated_tokens = []
        finished = jnp.zeros((batch_size,), dtype=bool)

        for step in range(max_gen_len):
            if jnp.all(finished):
                break

            rng, step_rng = jax.random.split(rng)

            # 采样当前 token
            next_token = sample_token(current_logits, step_rng, step,
                          eos_id=tokenizer.eos_token, tokenizer=tokenizer,
                          min_gen_len=3,
                          sampler="temperature", temperature=temperature, nucleus_p=top_p)
            # next_token = sample_token(current_logits, step_rng, step)

            generated_tokens.append(next_token)
            finished = jnp.logical_or(finished, next_token == tokenizer.eos_token)

            # 如果还需要继续，重新计算完整序列
            if step + 1 < max_gen_len and not jnp.all(finished):
                # 构造完整序列：prefix + 已生成的 tokens + 当前 token
                gen_so_far = jnp.stack(generated_tokens, axis=1)  # [batch, step+1]
                gen_embeddings = self.PaliGemma.llm(gen_so_far, method="embed")  # [batch, step+1, dim]

                # 合并 prefix + 生成的部分
                full_tokens = jnp.concatenate([prefix_tokens, gen_embeddings], axis=1)
                full_mask = jnp.concatenate([
                    prefix_mask,
                    jnp.ones((batch_size, gen_so_far.shape[1]), dtype=bool)
                ], axis=1)

                # AR mask：prefix 非自回归，生成部分自回归
                gen_ar_mask = jnp.ones((gen_so_far.shape[1],), dtype=bool)
                full_ar_mask = jnp.concatenate([prefix_ar_mask, gen_ar_mask], axis=0)

                # 计算注意力掩码和位置
                full_attn_mask = make_attn_mask(full_mask, full_ar_mask)
                positions = jnp.cumsum(full_mask, axis=1) - 1

                # 重新计算 logits
                (full_out, _), _ = self.PaliGemma.llm(
                    [full_tokens, None], mask=full_attn_mask, positions=positions, deterministic=True
                )

                # 提取最后位置的 logits
                current_logits = full_out[:, -1, :]

        # 5) 解码生成的文本
        if generated_tokens:
            generated_sequence = jnp.stack(generated_tokens, axis=1)

            decode_text = []
            for i in range(batch_size):
                tokens_i = generated_sequence[i]
                # 找到 EOS 位置
                eos_positions = jnp.where(tokens_i == tokenizer.eos_token)[0]
                if len(eos_positions) > 0:
                    valid_tokens = tokens_i[:eos_positions[0]]
                else:
                    valid_tokens = tokens_i

                decoded = tokenizer.decode(valid_tokens.tolist())
                decode_text.append(decoded)
        else:
            decode_text = [""] * batch_size

        return decode_text

        # # 3) 回环解码（使用原始 token ids，不是 embedding）
        # num_text_tokens = observation.tokenized_prompt.shape[1]
        # text_ids = np.array(observation.tokenized_prompt[0])  # 原始 token ids [T]
        # text_mask = np.array(observation.tokenized_prompt_mask[0])  # 对应的 mask [T]
        # valid_len = int(text_mask.sum())  # 只取有效部分
        # valid_text_ids = text_ids[:valid_len]
        # print("input text ids:", valid_text_ids.tolist())
        # print("input text (decoded):", tokenizer.decode(valid_text_ids.tolist()))
        #
        # # 4) 检查越界
        # mx = int(valid_text_ids.max())
        # assert mx < tokenizer.vocab_size, f"token id {mx} >= vocab_size {tokenizer.vocab_size}"
        #
        # # 5) 位置与掩码 sanity check
        # positions_dbg = jnp.cumsum(prefix_mask, axis=1) - 1
        # assert positions_dbg.dtype == jnp.int32 or positions_dbg.dtype == jnp.int64

        # # TODO 添加自回归生成文本token
        # prefix_len = int(prefix_tokens.shape[1])
        # total_max_len = prefix_len + max_gen_len
        #
        # generated_tokens = jnp.zeros((batch_size, total_max_len), dtype=jnp.int32)
        # cur_token = jnp.full((batch_size, 1), tokenizer.bos_token, dtype=jnp.int32)
        # finished = jnp.zeros((batch_size,), dtype=bool)
        #
        # for step in range(max_gen_len):
        #     # 构造 mask/positions
        #     # 只让当前 token attend 到所有 prefix+已生成 token
        #     cur_total_len = prefix_len + step + 1
        #     input_mask = jnp.arange(cur_total_len)[None, :] < cur_total_len
        #     mask_ar = jnp.ones((batch_size, cur_total_len), dtype=jnp.int32)
        #     attn_mask = make_attn_mask(input_mask, mask_ar)
        #     attn_mask = attn_mask[:, -1:,:]
        #
        #     pos = jnp.full((batch_size, 1), cur_total_len - 1, dtype=jnp.int32)
        #
        #     # cur_token_embedding = self.PaliGemma.llm(cur_token, method="embed")
        #     # logits, kv_cache = self.PaliGemma.llm([cur_token_embedding, None],
        #     #                                       mask=attn_mask,
        #     #                                       positions=pos,
        #     #                                       kv_cache=kv_cache,
        #     #                                       deterministic=True
        #     #                                       )
        #     # next_token = jnp.argmax(logits[0][:, -1, :], axis=-1)
        #     # generated_tokens = generated_tokens.at[:, cur_total_len-1].set(next_token)
        #     # finished = jnp.logical_or(finished, next_token == tokenizer.eos_token)
        #     # cur_token = next_token[:, None]
        #     # if jnp.all(finished):
        #     #     break
        #
        #     cur_token_embedding = self.PaliGemma.llm(cur_token, method="embed")
        #     logits, kv_cache = self.PaliGemma.llm([cur_token_embedding, None],
        #                                           mask=attn_mask,
        #                                           positions=pos,
        #                                           kv_cache=kv_cache,
        #                                           deterministic=True
        #                                           )
        #     last_logits = logits[0][:, -1, :]  # [b, vocab]
        #     # 可选屏蔽无效token（例如pad），避免采样到
        #     try:
        #         pad_id = self.tokenizer.pad_token
        #         last_logits = last_logits.at[:, pad_id].set(-1e30)
        #     except Exception:
        #         pass
        #
        #     next_token = jnp.argmax(last_logits, axis=-1)
        #     generated_tokens = generated_tokens.at[:, cur_total_len-1].set(next_token)
        #     finished = jnp.logical_or(finished, next_token == tokenizer.eos_token)
        #     cur_token = next_token[:, None]
        #     if jnp.all(finished):
        #         break
        #
        # # generated_tokens = jnp.stack(generated_tokens, axis=1)  # [batch, gen_len]
        # decode_text = [
        #     tokenizer.decode(tokens[:prefix_len+max_gen_len].tolist()) for tokens in generated_tokens
        # ]
        # return decode_text
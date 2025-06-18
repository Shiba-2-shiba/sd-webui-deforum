import torch
import math

from .uni_pc_fm import sample_unipc
from .wrapper import fm_wrapper
from .utils import repeat_to_batch_size


def flux_time_shift(t, mu=1.15, sigma=1.0):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def calculate_flux_mu(context_length, x1=256, y1=0.5, x2=4096, y2=1.15, exp_max=7.0):
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    mu = k * context_length + b
    mu = min(mu, math.log(exp_max))
    return mu


def get_flux_sigmas_from_mu(n, mu):
    sigmas = torch.linspace(1, 0, steps=n + 1)
    sigmas = flux_time_shift(sigmas, mu=mu)
    return sigmas


@torch.inference_mode()
def sample_hunyuan(
        transformer,
        sampler='unipc',
        initial_latent=None,
        concat_latent=None,
        strength=1.0,
        width=512,
        height=512,
        frames=16,
        real_guidance_scale=1.0,
        distilled_guidance_scale=6.0,
        guidance_rescale=0.0,
        shift=None,
        num_inference_steps=25,
        batch_size=None,
        generator=None,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        prompt_poolers=None,
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        negative_prompt_poolers=None,
        dtype=torch.bfloat16,
        device=None,
        negative_kwargs=None,
        callback=None,
        **kwargs,
):
    device = device or transformer.device

    # ★★★ ログ1: 関数の入力となる initial_latent の状態 ★★★
    if initial_latent is not None:
        print(f"\n[LOG 1] initial_latent received by sampler:"
              f"\n  - shape: {initial_latent.shape}, dtype: {initial_latent.dtype}"
              f"\n  - min: {initial_latent.min():.4f}, max: {initial_latent.max():.4f}, mean: {initial_latent.mean():.4f}\n")

    if batch_size is None:
        batch_size = int(prompt_embeds.shape[0])

    latents = torch.randn((batch_size, 16, (frames + 3) // 4, height // 8, width // 8), generator=generator, device=generator.device).to(device=device, dtype=torch.float32)

    # ★★★ ログ2: 生成された初期ノイズの状態 ★★★
    print(f"[LOG 2] Initial random noise 'latents' created:"
          f"\n  - shape: {latents.shape}, dtype: {latents.dtype}"
          f"\n  - min: {latents.min():.4f}, max: {latents.max():.4f}, mean: {latents.mean():.4f}\n")

    B, C, T, H, W = latents.shape
    seq_length = T * H * W // 4

    if shift is None:
        mu = calculate_flux_mu(seq_length, exp_max=7.0)
    else:
        mu = math.log(shift)

    sigmas = get_flux_sigmas_from_mu(num_inference_steps, mu).to(device)

    k_model = fm_wrapper(transformer)

    if initial_latent is not None:
        sigmas = sigmas * strength
        first_sigma = sigmas[0].to(device=device, dtype=torch.float32)
        initial_latent = initial_latent.to(device=device, dtype=torch.float32)
        latents = initial_latent.float() * (1.0 - first_sigma) + latents.float() * first_sigma

        # ★★★ ログ3: ノイズ混合後の latent の状態 ★★★
        print(f"[LOG 3] 'latents' after mixing with initial_latent (INPUT to UNet):"
              f"\n  - strength: {strength:.4f}, first_sigma: {first_sigma:.4f}"
              f"\n  - shape: {latents.shape}, dtype: {latents.dtype}"
              f"\n  - min: {latents.min():.4f}, max: {latents.max():.4f}, mean: {latents.mean():.4f}\n")

    if concat_latent is not None:
        concat_latent = concat_latent.to(latents)

    distilled_guidance = torch.tensor([distilled_guidance_scale * 1000.0] * batch_size).to(device=device, dtype=dtype)

    prompt_embeds = repeat_to_batch_size(prompt_embeds, batch_size)
    prompt_embeds_mask = repeat_to_batch_size(prompt_embeds_mask, batch_size)
    prompt_poolers = repeat_to_batch_size(prompt_poolers, batch_size)
    negative_prompt_embeds = repeat_to_batch_size(negative_prompt_embeds, batch_size)
    negative_prompt_embeds_mask = repeat_to_batch_size(negative_prompt_embeds_mask, batch_size)
    negative_prompt_poolers = repeat_to_batch_size(negative_prompt_poolers, batch_size)
    concat_latent = repeat_to_batch_size(concat_latent, batch_size)

    sampler_kwargs = dict(
        dtype=dtype,
        cfg_scale=real_guidance_scale,
        cfg_rescale=guidance_rescale,
        concat_latent=concat_latent,
        positive=dict(
            pooled_projections=prompt_poolers,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_embeds_mask,
            guidance=distilled_guidance,
            **kwargs,
        ),
        negative=dict(
            pooled_projections=negative_prompt_poolers,
            encoder_hidden_states=negative_prompt_embeds,
            encoder_attention_mask=negative_prompt_embeds_mask,
            guidance=distilled_guidance,
            **(kwargs if negative_kwargs is None else {**kwargs, **negative_kwargs}),
        )
    )

    if sampler == 'unipc':
        results = sample_unipc(k_model, latents, sigmas, extra_args=sampler_kwargs, disable=False, callback=callback)
    else:
        raise NotImplementedError(f'Sampler {sampler} is not supported.')

    # ★★★ ログ4: サンプラーからの最終出力 ★★★
    print(f"\n[LOG 4] Final 'results' from sampler:"
          f"\n  - shape: {results.shape}, dtype: {results.dtype}"
          f"\n  - min: {results.min():.4f}, max: {results.max():.4f}, mean: {results.mean():.4f}\n")
    
    return results

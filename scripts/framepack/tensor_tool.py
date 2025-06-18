# tensor_tool.py (tensor_processing.py連携 最終版)

import os
import torch
import numpy as np
import re
from PIL import Image

# tensor_processingからデコード関数をインポート
from . import tensor_processing

# Framepack F1のコア機能とヘルパー関数をインポート
from .k_diffusion_hunyuan import sample_hunyuan
from .utils import (
    crop_or_pad_yield_mask,
    resize_and_center_crop,
)
from .hunyuan import encode_prompt_conds, vae_encode
from .clip_vision import hf_clip_vision_encode
from .bucket_tools import find_nearest_bucket
from .vae_settings import apply_vae_settings

# メモリ管理ユーティリティ
from .memory import (
    cpu,
    gpu,
    load_model_as_complete,
    unload_complete_models,
    move_model_to_device_with_memory_preservation,
    offload_model_from_device_for_memory_preservation,
)

# スケジュール文字列から数値を抽出するヘルパー関数
def parse_schedule_string(schedule_str: str) -> float:
    """ "0: (1.23)" のような文字列から数値部分を抽出する """
    match = re.search(r'\((.*?)\)', schedule_str)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return 0.0
    return 0.0

@torch.no_grad()
def execute_generation(managers: dict, device, args, anim_args, video_args, framepack_f1_args, root):
    """
    Deforumから呼び出される動画生成のメイン関数。
    複数チャンク生成に対応するためループ処理を導入し、
    チャンク毎のデコード処理をtensor_processingモジュールに委譲する。
    """
    print("[tensor_tool] Starting multi-chunk video generation process...")

    # --- 1. マネージャーからモデルを取得 ---
    transformer_manager = managers["transformer"]
    text_encoder_manager = managers["text_encoder"]
    vae_manager = managers["vae"]
    image_encoder_manager = managers["image_encoder"]
    image_processor_manager = managers["image_processor"]
    tokenizer_manager = managers["tokenizers"]
    
    transformer_manager.ensure_transformer_state()
    text_encoder_manager.ensure_text_encoder_state()
    
    transformer = transformer_manager.get_transformer()
    text_encoder, text_encoder_2 = text_encoder_manager.get_text_encoders()
    vae = vae_manager.get_model()
    image_encoder = image_encoder_manager.get_model()
    image_processor = image_processor_manager.get_processor()
    tokenizer, tokenizer_2 = tokenizer_manager.get_tokenizers()
    
    high_vram = transformer_manager.current_state['high_vram']

    # --- 2. パラメータの準備 ---
    prompt = args.positive_prompts
    n_prompt = args.negative_prompts if hasattr(args, 'negative_prompts') else ""
    seed = args.seed
    steps = args.steps
    width, height = args.W, args.H

    strength = framepack_f1_args.f1_image_strength # ★★★ このように変更 ★★★
    cfg = parse_schedule_string(getattr(anim_args, 'cfg_scale_schedule', "0: (1.0)"))
    gs = parse_schedule_string(anim_args.distilled_cfg_scale_schedule)
    rs = getattr(framepack_f1_args, 'guidance_rescale', 0.0)
    latent_window_size = framepack_f1_args.f1_generation_latent_size
    
    # ループ制御用の変数を初期化
    total_frames_to_generate = anim_args.max_frames
    start_frame_idx = anim_args.extract_from_frame if hasattr(anim_args, 'extract_from_frame') else 0
    frame_idx = start_frame_idx
    saved_count = 0

    # --- 3. プロンプトエンコード ---
    print("[tensor_tool] Encoding prompts...")
    if not high_vram:
        load_model_as_complete(text_encoder, target_device=device)
        load_model_as_complete(text_encoder_2, target_device=device)

    prompt_embeds, prompt_poolers = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
    n_prompt_embeds, n_prompt_poolers = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

    prompt_embeds, prompt_embeds_mask = crop_or_pad_yield_mask(prompt_embeds, length=256)
    n_prompt_embeds, n_prompt_embeds_mask = crop_or_pad_yield_mask(n_prompt_embeds, length=256)
    
    if not high_vram:
        unload_complete_models(text_encoder, text_encoder_2)

    # --- 4. 初期画像の準備と最初のLatentの生成 ---
    print("[tensor_tool] Processing initial image for the first chunk...")
    init_image_path = args.init_image
    if not init_image_path or not os.path.exists(init_image_path):
        raise FileNotFoundError(f"Initial image not found at: {init_image_path}")

    input_image_pil = Image.open(init_image_path).convert("RGB")
    bucket_w, bucket_h = find_nearest_bucket(width, height)
    input_image_np = resize_and_center_crop(np.array(input_image_pil), bucket_w, bucket_h)

    if not high_vram: load_model_as_complete(vae, target_device=device)
    img_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1.0
    img_pt = img_pt.permute(2, 0, 1).unsqueeze(0).to(device).unsqueeze(2)
    current_latent = vae_encode(img_pt, vae)
    # 生成履歴を保持するhistory_latentsを初期化
    history_latents = torch.zeros(size=(1, 16, 16 + 2 + 1, bucket_h // 8, bucket_w // 8), dtype=torch.float32, device=cpu)
    history_latents = torch.cat([history_latents, current_latent.to(cpu)], dim=2)
    start_latent = current_latent.clone() # 開始latentを別途保持
    # ▲▲▲【追加ここまで】▲▲▲
    if not high_vram: unload_complete_models(vae)

    if not high_vram: load_model_as_complete(image_encoder, target_device=device)
    image_embeddings = hf_clip_vision_encode(input_image_np, image_processor, image_encoder).last_hidden_state
    if not high_vram: unload_complete_models(image_encoder)

    # --- 5. メイン生成ループ ---
    print(f"[tensor_tool] Starting generation loop for a total of {total_frames_to_generate} frames.")
    while frame_idx < total_frames_to_generate:
        chunk_num = (frame_idx // (int(latent_window_size * 4 - 3))) + 1
        print(f"\n--- Generating Chunk {chunk_num} (current frame: {frame_idx}) ---")

        if not high_vram:
            preserved_memory = getattr(framepack_f1_args, 'preserved_memory', 8.0)
            move_model_to_device_with_memory_preservation(transformer, target_device=device, preserved_memory_gb=preserved_memory)

        rnd = torch.Generator(device=device).manual_seed(seed + frame_idx)
        
        # --- history_latentsからコンテキストを生成 ---
        frames_to_generate_in_latent = int(latent_window_size * 4 - 3)
        effective_window_size = int(latent_window_size)
        indices = torch.arange(0, sum([1, 16, 2, 1, effective_window_size])).unsqueeze(0)
    
        clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = \
            indices.split([1, 16, 2, 1, effective_window_size], dim=1)
        clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

        clean_latents_4x, clean_latents_2x, clean_latents_1x = \
            history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)

        clean_latents = torch.cat([start_latent.cpu(), clean_latents_1x], dim=2)

        # 履歴コンテキストの影響をわずかに減衰させる（例: 5%減）
        # これにより、過去のフレームの残像効果を弱めることを狙う
        damping_factor = 1.0
        clean_latents = clean_latents * damping_factor
        print(f"[tensor_tool] Applied context damping with factor: {damping_factor}")


        # --- コンテキスト生成ここまで ---

        sampler_kwargs = dict(
            transformer=transformer, sampler="unipc", strength=strength, width=bucket_w, height=bucket_h,
            frames=frames_to_generate_in_latent, real_guidance_scale=cfg, distilled_guidance_scale=gs,
            guidance_rescale=rs, num_inference_steps=steps, generator=rnd,
            prompt_embeds=prompt_embeds.to(transformer.dtype), prompt_embeds_mask=prompt_embeds_mask,
            prompt_poolers=prompt_poolers.to(transformer.dtype), negative_prompt_embeds=n_prompt_embeds.to(transformer.dtype),
            negative_prompt_embeds_mask=n_prompt_embeds_mask, negative_prompt_poolers=n_prompt_poolers.to(transformer.dtype),
            image_embeddings=image_embeddings.to(transformer.dtype),
            initial_latent=current_latent,
            device=device, dtype=torch.bfloat16,
            latent_indices=latent_indices,
            clean_latents=clean_latents,
            clean_latent_indices=clean_latent_indices,
            clean_latents_2x=clean_latents_2x,
            clean_latent_2x_indices=clean_latent_2x_indices,
            clean_latents_4x=clean_latents_4x,
            clean_latent_4x_indices=clean_latent_4x_indices,
        )
        
        generated_latents = sample_hunyuan(**sampler_kwargs)
        print(f"[tensor_tool] Sampler generated {generated_latents.shape[2]} key latent frames.")

        history_latents = torch.cat([history_latents, generated_latents.to(cpu)], dim=2)
        current_latent = generated_latents[:, :, -1:, :, :].clone()

        if generated_latents.shape[2] != frames_to_generate_in_latent:
            generated_latents = torch.nn.functional.interpolate(
                generated_latents.to(torch.float32),
                size=(frames_to_generate_in_latent, generated_latents.shape[3], generated_latents.shape[4]),
                mode='nearest'
            ).to(generated_latents.dtype)
        
        print(f"[tensor_tool] Delegating decoding of chunk {chunk_num} to tensor_processing module...")
        if not high_vram: load_model_as_complete(vae, target_device=device)
        
        apply_vae_settings(vae)
        pixels = tensor_processing.process_latents(latents=generated_latents, vae=vae, use_vae_cache=True)
        
        if not high_vram: unload_complete_models(vae)

        frame_list = list(pixels.split(1, dim=2))
        for frame_tensor in frame_list:
            if frame_idx >= total_frames_to_generate:
                break
            
            frame_4d = frame_tensor.squeeze(2)
            resized_frame = torch.nn.functional.interpolate(frame_4d, size=(height, width), mode='bilinear', align_corners=False)
            
            frame_np = (resized_frame.squeeze(0).cpu() + 1.0) / 2.0
            frame_np = (frame_np.clamp(0, 1) * 255.0).permute(1, 2, 0).to(torch.float32).numpy().astype(np.uint8)

            filename = f"{root.timestring}_{frame_idx:09d}.png"
            filepath = os.path.join(args.outdir, filename)
            Image.fromarray(frame_np).save(filepath)
            
            frame_idx += 1
            saved_count += 1
        
        print(f"[tensor_tool] Chunk {chunk_num} processed. Total frames saved: {saved_count}/{total_frames_to_generate}")
    # --- 6. 処理完了と戻り値 ---
    print(f"\n[tensor_tool] Process complete. Total {saved_count} frames saved to {args.outdir}")
    # アーキテクチャ設計に合わせ、UI更新時の不整合を避けるため戻り値は常にNoneとする 
    return None

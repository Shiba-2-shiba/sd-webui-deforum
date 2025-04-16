# reforum/generate.py の修正版 v2 (sd-webui-forge-classic + ControlNet + AttributeError 修正)

import os
import math
import json
import itertools
import requests
import numexpr
import copy # copy モジュールをインポート
from PIL import Image
import numpy as np # numpy をインポート
import cv2 # cv2 をインポート

from modules import processing, sd_models, scripts # scripts をインポート
from modules.shared import sd_model, state, cmd_opts
from modules.processing import StableDiffusionProcessingTxt2Img # Txt2Img用クラスをインポート

# --- Deforum (reforum) 固有のインポート (既存) ---
from .deforum_controlnet import is_controlnet_enabled
from .prompt import split_weighted_subprompts, check_is_number
from .load_images import load_img, prepare_mask, check_mask_for_errors # load_img の引数を確認
from .webui_sd_pipeline import get_webui_sd_pipeline
from .rich import console # rich をインポート
from .defaults import get_samplers_list
from .general_utils import debug_print # デバッグ用

# --- Forge連携用ヘルパー関数のインポート ---
try:
    from .deforum_scripts_overrides import add_forge_script_to_deforum_run, initialise_forge_scripts
except ImportError:
    print("\n!!! [Deforum] Warning: deforum_scripts_overrides.py not found. Forge script integration (ControlNet, etc.) might not work correctly. !!!\n")
    def initialise_forge_scripts(p): pass
    def add_forge_script_to_deforum_run(p, script_title, script_args):
        print(f"[Deforum] Error: Tried to add script '{script_title}' but deforum_scripts_overrides is missing.")

# --- ヘルパー関数 (変更なし) ---
def load_mask_latent(mask_input, shape):
    if isinstance(mask_input, str):
        if mask_input.startswith('http://') or mask_input.startswith('https://'):
            mask_image = Image.open(requests.get(mask_input, stream=True).raw).convert('RGBA')
        else:
            mask_image = Image.open(mask_input).convert('RGBA')
    elif isinstance(mask_input, Image.Image):
        mask_image = mask_input
    else:
        raise Exception("mask_input must be a PIL image or a file name")

    mask_w_h = (shape[-1], shape[-2])
    mask = mask_image.resize(mask_w_h, resample=Image.LANCZOS)
    mask = mask.convert("L")
    return mask

def isJson(myjson):
    try:
        json.loads(myjson)
    except ValueError as e:
        return False
    return True

def pairwise_repl(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

# --- generate 関数 (変更なし) ---
def generate(args, keys, anim_args, loop_args, controlnet_args, root, frame=0, sampler_name=None, controlnet_units=None):
    if state.interrupted:
        return None

    if args.reroll_blank_frames == 'ignore':
        return generate_inner(args, keys, anim_args, loop_args, controlnet_args, root, frame, sampler_name, controlnet_units)

    image, caught_vae_exception = generate_with_nans_check(args, keys, anim_args, loop_args, controlnet_args, root, frame, sampler_name, controlnet_units)

    if caught_vae_exception or not image.getbbox():
        patience = args.reroll_patience
        print("Blank frame detected! If you don't have the NSFW filter enabled, this may be due to a glitch!")
        if args.reroll_blank_frames == 'reroll':
            while caught_vae_exception or not image.getbbox():
                print("Rerolling with +1 seed...")
                args.seed += 1
                image, caught_vae_exception = generate_with_nans_check(args, keys, anim_args, loop_args, controlnet_args, root, frame, sampler_name, controlnet_units)
                patience -= 1
                if patience == 0:
                    print("Rerolling with +1 seed failed for 10 iterations! Try setting webui's precision to 'full' and if it fails, please report this to the devs! Interrupting...")
                    state.interrupted = True
                    state.assign_current_image(image)
                    return None
        elif args.reroll_blank_frames == 'interrupt':
            print("Interrupting to save your eyes...")
            state.interrupted = True
            state.assign_current_image(image)
            return None
    return image

# --- generate_with_nans_check 関数 (変更なし) ---
def generate_with_nans_check(args, keys, anim_args, loop_args, controlnet_args, root, frame=0, sampler_name=None, controlnet_units=None):
    if cmd_opts.disable_nan_check:
        image = generate_inner(args, keys, anim_args, loop_args, controlnet_args, root, frame, sampler_name, controlnet_units)
    else:
        try:
            image = generate_inner(args, keys, anim_args, loop_args, controlnet_args, root, frame, sampler_name, controlnet_units)
        except Exception as e:
            if "A tensor with all NaNs was produced in VAE." in repr(e):
                print(e)
                return None, True
            else:
                raise e
    return image, False

# --- generate_inner 関数 (メイン処理、AttributeError修正) ---
def generate_inner(args, keys, anim_args, loop_args, controlnet_args, root, frame=0, sampler_name=None, controlnet_units=None):
    p = get_webui_sd_pipeline(args, root)
    p.prompt, p.negative_prompt = split_weighted_subprompts(args.prompt, frame, anim_args.max_frames)

    if not args.use_init and args.strength > 0 and args.strength_0_no_init:
        args.strength = 0
    processed = None
    mask_image = None
    init_image = None
    image_init0 = None # Looper用

    # --- Looper 関連処理 (変更なし) ---
    if loop_args.use_looper and anim_args.animation_mode in ['2D', '3D']:
        args.strength = loop_args.imageStrength
        tweeningFrames = loop_args.tweeningFrameSchedule
        blendFactor = .07
        colorCorrectionFactor = loop_args.colorCorrectionFactor
        jsonImages = json.loads(loop_args.imagesToKeyframe)
        parsedImages = {}
        frameToChoose = 0
        for key, value in jsonImages.items():
            if check_is_number(key):
                 parsedImages[key] = value
            else:
                 parsedImages[int(numexpr.evaluate(key))] = value
        framesToImageSwapOn = list(map(int, list(parsedImages.keys())))
        for swappingFrame in framesToImageSwapOn[1:]:
            frameToChoose += (frame >= int(swappingFrame))
        skipFrame = 25
        for fs, fe in pairwise_repl(framesToImageSwapOn):
            if fs <= frame <= fe:
                skipFrame = fe - fs
        if frame % skipFrame <= tweeningFrames:
            blendFactor = loop_args.blendFactorMax - loop_args.blendFactorSlope * math.cos((frame % tweeningFrames) / (tweeningFrames / 2))
        init_image2, _ = load_img(list(jsonImages.values())[frameToChoose],
                                  shape=(args.W, args.H),
                                  use_alpha_as_mask=args.use_alpha_as_mask) # load_img は box 引数を受け取らない想定
        image_init0 = list(jsonImages.values())[0]
    else:
        # 通常の初期画像パスを設定
        image_init0 = args.init_image
        # <<< 修正箇所 >>> init_image_box への参照を削除
        # image_init0_box = args.init_image_box # この行を削除

    # --- Sampler, Checkpoint 設定 (変更なし) ---
    available_samplers = get_samplers_list()
    if sampler_name is not None:
        if sampler_name in available_samplers.keys():
            p.sampler_name = available_samplers[sampler_name]
        else:
            raise RuntimeError(f"Sampler name '{sampler_name}' is invalid. Please check the available sampler list in the 'Run' tab")

    if args.checkpoint is not None:
        info = sd_models.get_closet_checkpoint_match(args.checkpoint)
        if info is None:
            raise RuntimeError(f"Unknown checkpoint: {args.checkpoint}")
        sd_models.reload_model_weights(info=info)

    # --- Init Image / Txt2Img パス分岐 ---
    if root.init_sample is not None:
        img = root.init_sample
        init_image = img
        if loop_args.use_looper and isJson(loop_args.imagesToKeyframe) and anim_args.animation_mode in ['2D', '3D']:
             init_image = Image.blend(init_image, init_image2, blendFactor)
             correction_colors = Image.blend(init_image, init_image2, colorCorrectionFactor)
             p.color_corrections = [processing.setup_color_correction(correction_colors)]

    elif (loop_args.use_looper and anim_args.animation_mode in ['2D', '3D']) or (args.use_init and ((args.init_image != None and args.init_image != ''))): # box への参照は削除
        # <<< 修正箇所 >>> load_img 呼び出しから box 引数を削除
        init_image, mask_image = load_img(image_init0, # box 引数を削除
                                          shape=(args.W, args.H),
                                          use_alpha_as_mask=args.use_alpha_as_mask)
    else:
        # --- Pure Txt2Img Path ---
        if anim_args.animation_mode != 'Interpolation':
            print(f"Not using an init image (doing pure txt2img)")

        p_txt = StableDiffusionProcessingTxt2Img(
            sd_model=sd_model,
            outpath_samples=root.tmp_deforum_run_duplicated_folder,
            outpath_grids=root.tmp_deforum_run_duplicated_folder,
            prompt=p.prompt,
            styles=p.styles,
            negative_prompt=p.negative_prompt,
            seed=p.seed,
            subseed=p.subseed,
            subseed_strength=p.subseed_strength,
            seed_resize_from_h=p.seed_resize_from_h,
            seed_resize_from_w=p.seed_resize_from_w,
            sampler_name=p.sampler_name,
            batch_size=p.batch_size,
            n_iter=p.n_iter,
            steps=p.steps,
            cfg_scale=p.cfg_scale,
            width=p.width,
            height=p.height,
            restore_faces=p.restore_faces,
            tiling=p.tiling,
            enable_hr=False,
            denoising_strength=0
        )

        print_combined_table(args, anim_args, p_txt, keys, frame)

        initialise_forge_scripts(p_txt)

        if is_controlnet_enabled(controlnet_args) and controlnet_units:
            add_forge_script_to_deforum_run(p_txt, "ControlNet", controlnet_units)
            print(f"[Deforum Txt2Img] Added {len(controlnet_units)} ControlNet units via add_forge_script_to_deforum_run.")
        elif is_controlnet_enabled(controlnet_args) and not controlnet_units:
             print(f"[Deforum Txt2Img Warning] ControlNet is enabled but no valid units were generated for frame {frame}. ControlNet will not be applied.")

        processed = processing.process_images(p_txt)


    # --- Img2Img Path または Txt2Img 完了後 ---
    if processed is None: # Img2Img パス
        if args.use_mask:
            mask_image = args.mask_image
            mask = prepare_mask(args.mask_file if mask_image is None else mask_image,
                                (args.W, args.H),
                                args.mask_contrast_adjust,
                                args.mask_brightness_adjust)
            p.inpainting_mask_invert = args.invert_mask
            p.inpainting_fill = args.fill
            p.inpaint_full_res = args.full_res_mask
            p.inpaint_full_res_padding = args.full_res_mask_padding
            mask = check_mask_for_errors(mask, args.invert_mask)
            root.noise_mask = mask
        else:
            mask = None
            root.noise_mask = None

        assert not ((mask is not None and args.use_mask and args.overlay_mask) and (
                root.init_sample is None and init_image is None)), "Need an init image when use_mask == True and overlay_mask == True"

        p.init_images = [init_image]
        p.image_mask = mask
        p.image_cfg_scale = args.pix2pix_img_cfg_scale

        print_combined_table(args, anim_args, p, keys, frame)

        initialise_forge_scripts(p)

        if is_controlnet_enabled(controlnet_args) and controlnet_units:
            add_forge_script_to_deforum_run(p, "ControlNet", controlnet_units)
            print(f"[Deforum Img2Img] Added {len(controlnet_units)} ControlNet units via add_forge_script_to_deforum_run.")
        elif is_controlnet_enabled(controlnet_args) and not controlnet_units:
             print(f"[Deforum Img2Img Warning] ControlNet is enabled but no valid units were generated for frame {frame}. ControlNet will not be applied.")
             if hasattr(p, 'scripts') and hasattr(p.scripts, 'scripts'):
                 print("[Deforum Debug] Available scripts in p.scripts.scripts:")
                 for idx_debug, script_obj_debug in enumerate(p.scripts.scripts):
                     title_debug = "N/A"
                     if hasattr(script_obj_debug, 'title') and callable(script_obj_debug.title):
                         title_debug = script_obj_debug.title()
                     print(f"  - Index {idx_debug}: Title='{title_debug}', Class='{type(script_obj_debug).__name__}'")

        processed = processing.process_images(p)

    # --- Results Handling ---
    if root.initial_info is None and hasattr(processed, 'info'):
        root.initial_info = processed.info

    if root.first_frame is None and hasattr(processed, 'images') and processed.images:
        root.first_frame = processed.images[0]

    results = None
    if hasattr(processed, 'images') and processed.images:
         results = processed.images[0]
    elif hasattr(processed, 'image'):
         results = processed.image
    else:
         print(f"[Deforum Warning] processing.process_images did not return expected images for frame {frame}.")
         return None

    return results


# --- print_combined_table 関数 (変更なし) ---
def print_combined_table(args, anim_args, p, keys, frame_idx):
    from rich.table import Table
    from rich import box

    table = Table(padding=0, box=box.ROUNDED)

    field_names1 = ["Steps", "CFG"]
    if anim_args.animation_mode != 'Interpolation':
        field_names1.append("Denoise")
    field_names1 += ["Subseed", "Subs. str"] * (anim_args.enable_subseed_scheduling)
    field_names1 += ["Sampler"] * anim_args.enable_sampler_scheduling
    field_names1 += ["Checkpoint"] * anim_args.enable_checkpoint_scheduling

    for field_name in field_names1:
        table.add_column(field_name, justify="center")

    steps_str = str(getattr(p, 'steps', 'N/A'))
    cfg_scale_str = str(getattr(p, 'cfg_scale', 'N/A'))
    denoise_str = "None"
    if hasattr(p, 'denoising_strength') and getattr(p, 'denoising_strength', None) is not None:
         denoise_str = f"{p.denoising_strength:.5g}"

    rows1 = [steps_str, cfg_scale_str]
    if anim_args.animation_mode != 'Interpolation':
        rows1.append(denoise_str)

    subseed_str = str(getattr(p, 'subseed', 'N/A'))
    subseed_strength_str = "N/A"
    if hasattr(p, 'subseed_strength'):
         subseed_strength_str = f"{p.subseed_strength:.5g}"

    rows1 += [subseed_str, subseed_strength_str] * anim_args.enable_subseed_scheduling
    rows1 += [getattr(p, 'sampler_name', 'N/A')] * anim_args.enable_sampler_scheduling
    rows1 += [str(args.checkpoint)] * anim_args.enable_checkpoint_scheduling

    rows2 = []
    if anim_args.animation_mode not in ['Video Input', 'Interpolation']:
        if anim_args.animation_mode == '2D':
            field_names2 = ["Angle", "Zoom"]
        else:
            field_names2 = []
        field_names2 += ["Tr X", "Tr Y"]
        if anim_args.animation_mode == '3D':
            field_names2 += ["Tr Z", "Ro X", "Ro Y", "Ro Z"]
            if anim_args.aspect_ratio_schedule.replace(" ", "") != '0:(1)':
                field_names2 += ["Asp. Ratio"]
        if anim_args.enable_perspective_flip:
            field_names2 += ["Pf T", "Pf P", "Pf G", "Pf F"]

        for field_name in field_names2:
            table.add_column(field_name, justify="center")

        def get_key_value(series, idx, default="N/A"):
            try:
                if series is not None and idx < len(series):
                    value = series[idx]
                    return f"{value:.5g}" if value is not None else default
                else:
                    return default
            except (IndexError, TypeError):
                return default

        if anim_args.animation_mode == '2D':
            rows2 += [get_key_value(keys.angle_series, frame_idx), get_key_value(keys.zoom_series, frame_idx)]
        rows2 += [get_key_value(keys.translation_x_series, frame_idx), get_key_value(keys.translation_y_series, frame_idx)]

        if anim_args.animation_mode == '3D':
            rows2 += [get_key_value(keys.translation_z_series, frame_idx),
                      get_key_value(keys.rotation_3d_x_series, frame_idx),
                      get_key_value(keys.rotation_3d_y_series, frame_idx),
                      get_key_value(keys.rotation_3d_z_series, frame_idx)]
            if anim_args.aspect_ratio_schedule.replace(" ", "") != '0:(1)':
                rows2 += [get_key_value(keys.aspect_ratio_series, frame_idx)]
        if anim_args.enable_perspective_flip:
            rows2 += [get_key_value(keys.perspective_flip_theta_series, frame_idx),
                      get_key_value(keys.perspective_flip_phi_series, frame_idx),
                      get_key_value(keys.perspective_flip_gamma_series, frame_idx),
                      get_key_value(keys.perspective_flip_fv_series, frame_idx)]

    table.add_row(*rows1, *rows2)
    console.print(table)

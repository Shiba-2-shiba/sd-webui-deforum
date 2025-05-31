# [source: 1] # [source: 122]
import os
import pandas as pd
import cv2
import numpy as np
import numexpr
import gc
import time
import math

from PIL import Image
from .generate import generate, isJson
from .noise import add_noise
from .animation_key_frames import DeforumAnimKeys, LooperAnimKeys
# [source: 122] からのインポート
from .video_audio_utilities import get_frame_name, get_next_frame, extract_video_init_without_hybrid, extract_video_for_cc_path
from .depth import DepthModel
from .parseq_adapter import ParseqAnimKeys
from .seed import next_seed
from .image_sharpening import unsharp_mask
from .load_images import get_mask, load_img, get_mask_from_file
from .composable_masks import compose_mask_with_check
from .settings import save_settings_from_animation_run
# --- ControlNet関連のインポート修正 ---
# [source: 278], [source: 265] 検討内容に基づき get_deforum_controlnet_units を追加
from .deforum_controlnet import unpack_controlnet_vids, is_controlnet_enabled, get_deforum_controlnet_units
# ------------------------------------
from .subtitle_handler import init_srt_file, write_frame_subtitle, format_animation_params
from .prompt import prepare_prompt

# reallybigname - dev note: I put this block together for easy reloading
# [source: 2] stuff I work on a lot with importlib
from .animation import anim_frame_warp
from .cadence import build_turbo_dict, cadence_animation_frames, print_cadence_msg, cadence_save
from .cadence_morpho import cadence_morpho
# [source: 123] からのインポート
from .cadence_temporal_flow import cadence_temporal_flow
from .cadence_hybrid_comp import cadence_hybrid_comp
from .cadence_hybrid_motion import cadence_hybrid_motion
from .cadence_flow import cadence_flow
from .colors import color_coherence, get_cc_sample_from_video_frame, get_cc_sample_from_image_path
from .composite import make_composite_with_conform
from .displayer import IntervalPreviewDisplayer
from .generation import optical_flow_generation, redo_generation
from .history import ThingHistory
from .hybrid_video import hybrid_generation
from .hybrid_compositing import hybrid_composite
from .hybrid_render import hybrid_motion, get_default_motion
from .image_functions import bgr2gray_bgr, bgr_np2pil_rgb
from .masks import do_overlay_mask
from .morphological import image_morphological_flow_transform
from .resume import get_resume_vars
from .temporal_flow import do_temporal_flow
from .updown_scale import updown_scale_to_integer

from modules.shared import opts, cmd_opts, state, sd_model
from modules import devices, sd_hijack
# lowvram = False # この行は重複しているように見えるのでコメントアウトまたは削除を検討
# lowvram = False # この行も同様
from .RAFT import RAFT

# IN PROGRESS
# [source: 3] START reallybigname dev code to reload stuff (to be removed)
# import sys
# import importlib
# ... (importlib.reload部分はコメントアウトまたは削除を推奨) ...
# END

def render_animation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root):

    #_/‾𝓐𝓷𝓲𝓶𝓪𝓽𝓲𝓸𝓷‾𝓢𝓮𝓽𝓾𝓹‾‾‾‾‾\_____________

    # [source: 4] # [source: 125] からの初期設定
    live_previews_enable_store = opts.live_previews_enable
    disp = IntervalPreviewDisplayer(frames_per_cycle=anim_args.diffusion_cadence)

    if opts.data.get("deforum_save_gen_info_as_srt", False):
        srt_filename = os.path.join(args.outdir, f"{root.timestring}.srt")
        srt_frame_duration = init_srt_file(srt_filename, video_args.fps)
    else:
        srt_filename = None
        # [source: 4]
        srt_frame_duration = None

    mode = anim_args.animation_mode
    # [source: 126] からの初期設定
    dimensions = (args.W, args.H)
    updown_scale = updown_scale_to_integer(anim_args.updown_scale)
    inputframes_path = os.path.join(args.outdir, 'inputframes')
    maskframes_path = os.path.join(args.outdir, 'maskframes')
    hybrid_cn_paths = []
    inputfiles = None
    # [source: 5] cn_files は不要になったため削除
    hybridframes_path = None
    cc_vid_folder = None # Initialize to prevent UnboundLocalError

    # --- ControlNet ビデオ展開 ---
    # ControlNetが有効な場合にのみ実行
    # [source: 5]
    if controlnet_args and is_controlnet_enabled(controlnet_args):
        unpack_controlnet_vids(args, anim_args, controlnet_args) # [source: 52] の呼び出し元は修正済みdefoum_controlnet.pyにある前提

    # [source: 127] 以降のモード別設定
    if mode in ['2D', '3D']:
        hybrid_overlay = None # Initialize here
        if anim_args.hybrid_composite != 'None' or anim_args.hybrid_motion != 'None': # [source: 128]
            # [source: 6] hybrid_generation は cn_files を返さなくなった想定
            args, anim_args, inputfiles, hybridframes_path = hybrid_generation(args, anim_args, inputframes_path, hybrid_cn_paths, root)
            # [source: 6]
            cc_vid_folder, cc_vid_path = 'inputframes', anim_args.video_init_path
        elif anim_args.color_coherence != 'None' and anim_args.color_coherence_source == 'Video Init': # [source: 129]
             cc_vid_folder, cc_vid_path = extract_video_init_without_hybrid(args.outdir, anim_args, inputframes_path)

        save_cc_mix = False # Set to True to enable saving mixed samples
        if save_cc_mix:
            # [source: 7]
            cc_mix_outdir = os.path.join(args.outdir, "ccmixed")
            # [source: 7]
            print(f"Saving color coherence mixed sample frames to:\n{cc_mix_outdir}") # [source: 130]
            if not os.path.exists(cc_mix_outdir):
                os.makedirs(cc_mix_outdir)
        else:
            cc_mix_outdir = None

        if anim_args.color_coherence_source == 'Video Path':
            # [source: 8] # [source: 131]
            # [source: 8]
             cc_vid_folder, cc_vid_path = extract_video_for_cc_path(args.outdir, anim_args, 'ccvideoframes')

        if loop_args.use_looper:
            print("Using Guided Images mode: seed_behavior will be set to 'schedule' and 'strength_0_no_init' to False")
            if args.strength == 0:
                raise RuntimeError("Strength needs to be greater than 0 in Init tab")
                # [source: 9]
            args.strength_0_no_init = False
            # [source: 9] # [source: 132]
            args.seed_behavior = "schedule"
            if not isJson(loop_args.init_images):
                raise RuntimeError("The images set for use with keyframe-guidance are not in a proper JSON format")

    # [source: 10]
    color_matcher = None # Initialize
    if anim_args.color_coherence != 'None':
        color_matcher = color_coherence() # [source: 133]

    use_parseq = parseq_args.parseq_manifest is not None and parseq_args.parseq_manifest.strip()

    # --- Initialize Keys and Parseq Adapter ---
    # [source: 10]
    if use_parseq:
        # [source: 52] Parseq Adapter を初期化 (ControlNetユニット取得にも使用)
        parseq_adapter = ParseqAnimKeys(parseq_args, anim_args, video_args)
        keys = parseq_adapter # keys として parseq_adapter を使用
        # [source: 11]
    else:
        parseq_adapter = None # Parseqを使用しない場合はNoneに設定
        keys = DeforumAnimKeys(anim_args, args.seed)
    # ------------------------------------------

    loopSchedulesAndData = LooperAnimKeys(loop_args, anim_args, args.seed)

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Saving animation frames to:\n{args.outdir}")

    if anim_args.cadence_save_turbo_frames:
        outdir_turbo = os.path.join(args.outdir, "turbo") # [source: 134]
        print(f"Saving turbo frames to:\n{outdir_turbo}")
        if not os.path.exists(outdir_turbo):
            # [source: 12]
            os.makedirs(outdir_turbo)
    else:
        outdir_turbo = None # Initialize

    save_settings_from_animation_run(args, anim_args, parseq_args, loop_args, controlnet_args, video_args, root)

    # [source: 11]
    turbo_steps = 1 if mode == 'Video Input' else int(anim_args.diffusion_cadence)

    # [source: 135] からの変数初期化 (修正なし)
    frame_idx = 0
    cc_sample = None
    prev_img = None
    prev_motion = get_default_motion(anim_args.hybrid_motion, dimensions)
    turbo_prev_image = None
    turbo_next_image = None
    # [source: 13]
    turbo = {} # 関数スコープで初期化
    matrix_flow = None

    return_conform_iterations_debug = True # Set to False for production?
    # [source: 14] # [source: 136]
    return_conform_iterations = anim_args.hybrid_comp_save_extra_frames and return_conform_iterations_debug

    if anim_args.temporal_flow == "None":
        img_history_max_states = 1 + turbo_steps
    else:
        # [source: 12]
        # Ensure temporal_flow_target_frame_schedule_series is available and not empty
        # target_frames = keys.temporal_flow_target_frame_schedule_series if hasattr(keys, 'temporal_flow_target_frame_schedule_series') and keys.temporal_flow_target_frame_schedule_series else [0]
        # img_history_max_states = 1 + (max(target_frames) + turbo_steps)
        target_frames_for_max = [0] # デフォルト値を設定
        if hasattr(keys, 'temporal_flow_target_frame_schedule_series'):
            schedule_series = keys.temporal_flow_target_frame_schedule_series
            # Series が None でなく、かつ空でないことを確認
            if schedule_series is not None and not schedule_series.empty:
                schedule_list = schedule_series.tolist()
                # リストが空でないことも確認
                if schedule_list:
                    target_frames_for_max = schedule_list
        # max() はリストに対して適用する
        img_history_max_states = 1 + (max(target_frames_for_max) + turbo_steps)

    # [source: 137] からの履歴初期化
    # [source: 15]
    img_history = ThingHistory(max_states=img_history_max_states)
    turbo_history = ThingHistory(max_states=1)
    hybrid_motion_history = ThingHistory(max_states=turbo_steps + 1)
    # [source: 138]
    z_translation_history = ThingHistory(max_states=25)

    if anim_args.resume_from_timestring:
        root.timestring = anim_args.resume_timestring
        frame_idx, prev_img, img_history = get_resume_vars(args.outdir, anim_args.resume_timestring, turbo_steps, img_history)

    # [source: 139] からのParseq設定 (修正なし)
    if use_parseq:
        # [source: 13]
        anim_args.flip_2d_perspective = True

    if use_parseq and hasattr(keys, 'manages_prompts') and keys.manages_prompts():
        # [source: 16]
        prompt_series = keys.prompts
    else:
        prompt_series = pd.Series([np.nan for a in range(anim_args.max_frames + 1)])
        if hasattr(root, 'animation_prompts'):
            for i, prompt in root.animation_prompts.items():
                try:
                    if str(i).isdigit(): # [source: 140]
                        # [source: 17]
                        idx = int(i)
                    else:
                        # [source: 14]
                        idx = int(numexpr.evaluate(str(i))) # Ensure i is string
                        # [source: 18]
                    if 0 <= idx < len(prompt_series):
                        prompt_series[idx] = prompt
                    else:
                        print(f"Warning: Prompt index {i} evaluated to {idx}, which is out of bounds for prompt_series (length {len(prompt_series)}).")
                        # [source: 19]
                except Exception as e:
                    print(f"Warning: Could not evaluate prompt index '{i}': {e}")
        prompt_series = prompt_series.ffill().bfill()


    predict_depths = (mode == '3D' and anim_args.use_depth_warping) or anim_args.save_depth_maps
    # [source: 141]
    predict_depths_for_hybrid = anim_args.hybrid_composite != 'None' and anim_args.hybrid_comp_mask_type in ['Depth', 'Video Depth']
    # [source: 20]
    depth_model = None # Initialize
    if predict_depths or predict_depths_for_hybrid:
        keep_in_vram = opts.data.get("deforum_keep_3d_models_in_vram", False)
        # --- エラー箇所修正 ---
        # device = ('cpu' if cmd_opts.lowvram or cmd_opts.medvram else root.device) # 修正前
        device = ('cpu' if getattr(cmd_opts, 'lowvram', False) or getattr(cmd_opts, 'medvram', False) else root.device) # 修正後
        # --------------------
        try:
            depth_model = DepthModel(root.models_path, device, root.half_precision, keep_in_vram=keep_in_vram, depth_algorithm=anim_args.depth_algorithm,
                                     # [source: 21]    # [source: 15]
                                     Width=args.W, Height=args.H,
                                     midas_weight=anim_args.midas_weight)
        except Exception as e:
            # [source: 22]
            print(f"Error initializing DepthModel: {e}")
            # [source: 142]
            depth_model = None
            anim_args.save_depth_maps = False
            anim_args.use_depth_warping = False # Disable warping if model fails
    else:
        anim_args.save_depth_maps = False

    raft_model = None
    load_raft = (anim_args.optical_flow_cadence == "RAFT" and int(anim_args.diffusion_cadence) > 1) or \
                (anim_args.hybrid_flow_method == "RAFT" and anim_args.hybrid_motion == "Optical Flow") or \
                (anim_args.hybrid_composite != "None" and anim_args.hybrid_comp_conform_method != "None") or \
                (anim_args.loop_comp_conform_method == "RAFT" and hasattr(keys, 'loop_comp_type_schedule_series') and any(x.lower() != 'none' for x in keys.loop_comp_type_schedule_series)) or \
                (anim_args.optical_flow_redo_generation == "RAFT") or \
                (anim_args.temporal_flow == "RAFT") or \
                (anim_args.morpho_flow == "RAFT") # [source: 143]
                # [source: 23] # [source: 24]
    if load_raft:
        print("Loading RAFT model...")
        try:
            raft_model = RAFT() # [source: 144]
        except Exception as e:
            # [source: 25]
            print(f"Error initializing RAFT model: {e}")
            raft_model = None # Ensure it's None if loading fails

    mask_vals = {}
    noise_mask_vals = {}
    mask_vals['everywhere'] = Image.new('1', dimensions, 1)
    noise_mask_vals['everywhere'] = Image.new('1', dimensions, 1)
    mask_image = None

    if args.use_init and args.init_image is not None and args.init_image != '':
        try:
            # [source: 26] [source: 18]
            _, mask_image = load_img(args.init_image, shape=dimensions, use_alpha_as_mask=args.use_alpha_as_mask) # [source: 145]
            if mask_image:
                mask_vals['video_mask'] = mask_image
                noise_mask_vals['video_mask'] = mask_image
        except Exception as e:
            print(f"Error loading init image mask: {e}")

    # [source: 27]
    # Mask handling needs to be done *inside* the loop for video masks
    # [source: 19] Removed initial loading of video mask here

    if anim_args.color_coherence_source in ['Image Path']:
        # [source: 147]
        cc_sample = get_cc_sample_from_image_path(anim_args.color_coherence_image_path, dimensions)

    state.job_count = anim_args.max_frames+1

    #_/‾𝓕𝓻𝓪𝓶𝓮‾𝓛𝓸𝓸𝓹‾𝓢𝓽𝓪𝓻𝓽‾‾‾‾‾\____________

    while frame_idx <= anim_args.max_frames:
        # Ensure frame_idx is within the bounds of the series
        if frame_idx >= len(keys.noise_schedule_series):
            # [source: 28]
            print(f"Warning: frame_idx {frame_idx} exceeds schedule length {len(keys.noise_schedule_series)}. Using last value.")
            # [source: 29]
            effective_frame_idx = len(keys.noise_schedule_series) - 1
        else:
            effective_frame_idx = frame_idx

        if turbo_steps == 1:
            state.job = f"frame {frame_idx + 1}/{anim_args.max_frames+1}"
            # [source: 20]
            state.job_no = frame_idx + 1 # [source: 30] [source: 148]
            if state.interrupted: # Check for interruption
                 print("\n** Interrupted by user **")
                 break
            if state.skipped:
                print("\n** PAUSED **")
                # [source: 31]
                state.skipped = False
                while not state.skipped:
                    time.sleep(0.1)
                    if state.interrupted: # Allow interruption while paused
                        print("\n** Interrupted by user while paused **")
                        # [source: 32]
                        break # Exit inner loop
                if state.interrupted:
                    break # Exit outer loop
                # [source: 21]
                # [source: 33]
                print("** RESUMING **") # [source: 149]

        # [source: 149] からのキーフレーム値取得 (Using effective_frame_idx)
        noise = keys.noise_schedule_series[effective_frame_idx]
        strength = keys.strength_schedule_series[effective_frame_idx]
        scale = keys.cfg_scale_schedule_series[effective_frame_idx]
        contrast = keys.contrast_schedule_series[effective_frame_idx]
        kernel = int(keys.kernel_schedule_series[effective_frame_idx])
        sigma = keys.sigma_schedule_series[effective_frame_idx]
        amount = keys.amount_schedule_series[effective_frame_idx]
        # [source: 34] # [source: 22]
        threshold = keys.threshold_schedule_series[effective_frame_idx]
        # [source: 150]
        cc_alpha = keys.color_coherence_alpha_schedule_series[effective_frame_idx]
        loop_comp_type = keys.loop_comp_type_schedule_series[effective_frame_idx]
        loop_comp_alpha = keys.loop_comp_alpha_schedule_series[effective_frame_idx]
        loop_comp_conform = keys.loop_comp_conform_schedule_series[effective_frame_idx]
        hybrid_comp_conform = keys.hybrid_comp_conform_schedule_series[effective_frame_idx]
        hybrid_flow_factor = keys.hybrid_flow_factor_schedule_series[effective_frame_idx]
        temporal_flow_factor = keys.temporal_flow_factor_schedule_series[effective_frame_idx]
        temporal_flow_motion_stabilizer_factor = keys.temporal_flow_motion_stabilizer_factor_schedule_series[effective_frame_idx]
        # [source: 35] # [source: 23]
        temporal_flow_rotation_stabilizer_factor = keys.temporal_flow_rotation_stabilizer_factor_schedule_series[effective_frame_idx]
        # [source: 151]
        temporal_flow_target_frame = math.floor(keys.temporal_flow_target_frame_schedule_series[effective_frame_idx])
        morpho = keys.morpho_schedule_series[effective_frame_idx]
        morpho_iterations = int(keys.morpho_iterations_schedule_series[effective_frame_idx] // 1)
        morpho_flow_factor = keys.morpho_flow_factor_schedule_series[effective_frame_idx]
        redo_flow_factor = keys.redo_flow_factor_schedule_series[effective_frame_idx]
        hybrid_comp_schedules = {
            # [source: 36]
            "alpha": keys.hybrid_comp_alpha_schedule_series[effective_frame_idx],
            "mask_alpha": keys.hybrid_comp_mask_alpha_schedule_series[effective_frame_idx],
            # [source: 24]
            "mask_contrast": keys.hybrid_comp_mask_contrast_schedule_series[effective_frame_idx],
            # [source: 152]
            "mask_auto_contrast_cutoff_low": keys.hybrid_comp_mask_auto_contrast_cutoff_low_schedule_series[effective_frame_idx],
            "mask_auto_contrast_cutoff_high": keys.hybrid_comp_mask_auto_contrast_cutoff_high_schedule_series[effective_frame_idx]
        }

        # [source: 37]
        scheduled_sampler_name = None
        scheduled_clipskip = None
        scheduled_noise_multiplier = None
        scheduled_ddim_eta = None
        scheduled_ancestral_eta = None
        # [source: 25]
        depth = None
        # [source: 153]
        mask_seq = None
        noise_mask_seq = None

        # [source: 38]
        # Use effective_frame_idx for schedule checks
        if anim_args.enable_steps_scheduling and keys.steps_schedule_series[effective_frame_idx] is not None:
            args.steps = int(keys.steps_schedule_series[effective_frame_idx])
        if anim_args.enable_sampler_scheduling and keys.sampler_schedule_series[effective_frame_idx] is not None:
            scheduled_sampler_name = keys.sampler_schedule_series[effective_frame_idx].casefold()
        # [source: 154]
        # [source: 26]
        if anim_args.enable_clipskip_scheduling and keys.clipskip_schedule_series[effective_frame_idx] is not None:
            # [source: 39]
            scheduled_clipskip = int(keys.clipskip_schedule_series[effective_frame_idx])
        if anim_args.enable_noise_multiplier_scheduling and keys.noise_multiplier_schedule_series[effective_frame_idx] is not None:
            scheduled_noise_multiplier = float(keys.noise_multiplier_schedule_series[effective_frame_idx])
        if anim_args.enable_ddim_eta_scheduling and keys.ddim_eta_schedule_series[effective_frame_idx] is not None:
            scheduled_ddim_eta = float(keys.ddim_eta_schedule_series[effective_frame_idx])
        # [source: 27] # [source: 155]
        if anim_args.enable_ancestral_eta_scheduling and keys.ancestral_eta_schedule_series[effective_frame_idx] is not None:
            # [source: 40]
            scheduled_ancestral_eta = float(keys.ancestral_eta_schedule_series[effective_frame_idx])
        if args.use_mask and keys.mask_schedule_series[effective_frame_idx] is not None:
            mask_seq = keys.mask_schedule_series[effective_frame_idx]
        if anim_args.use_noise_mask and keys.noise_mask_schedule_series[effective_frame_idx] is not None:
            noise_mask_seq = keys.noise_mask_schedule_series[effective_frame_idx]
        if args.use_mask and not anim_args.use_noise_mask:
            noise_mask_seq = mask_seq

        # --- Video Mask Loading (Inside Loop) ---
        # [source: 41]
        if anim_args.use_mask_video:
            # [source: 19] # [source: 146]
            mask_frame_path = get_next_frame(args.outdir, anim_args.video_mask_path, frame_idx, True)
            current_video_mask = get_mask_from_file(mask_frame_path, args)
            if current_video_mask:
                mask_vals['video_mask'] = current_video_mask
                # [source: 42]
                # If noise mask should also follow video mask and use_noise_mask is not True
                if not anim_args.use_noise_mask or noise_mask_seq is None or noise_mask_seq == 'video_mask':
                     noise_mask_vals['video_mask'] = current_video_mask
                     # If use_noise_mask is true, ensure noise_mask_seq uses 'video_mask'
                     # [source: 43]
                     if anim_args.use_noise_mask and (noise_mask_seq is None or noise_mask_seq != 'video_mask'):
                         print(f"Frame {frame_idx}: Forcing noise mask to follow video mask.")
                         noise_mask_seq = 'video_mask'
            # [source: 44]
            else:
                print(f"Warning: Could not load video mask frame: {mask_frame_path}")
                # Fallback to 'everywhere' if mask loading fails?
                # [source: 45]
                mask_vals['video_mask'] = mask_vals['everywhere']
                noise_mask_vals['video_mask'] = noise_mask_vals['everywhere']
        # ----------------------------------------

        # --- エラー箇所周辺 ---
        # if mode == '3D' and (cmd_opts.lowvram or cmd_opts.medvram): # 修正前
        if mode == '3D' and (getattr(cmd_opts, 'lowvram', False) or getattr(cmd_opts, 'medvram', False)): # 修正後
        # --------------------
            # [source: 28] # [source: 156]
             # lowvram.send_everything_to_cpu() # lowvram モジュール/オブジェクトが未定義の可能性
             # sd_hijack.model_hijack.undo_hijack(sd_model)
             devices.torch_gc()
             if depth_model and hasattr(depth_model, 'to'): depth_model.to(root.device)


        # [source: 157]
        if anim_args.color_coherence != 'None' and anim_args.color_coherence_source in ['Video Init', 'Video Path'] and cc_alpha > 0 and cc_vid_folder is not None:
            # [source: 29]
            cc_sample = get_cc_sample_from_video_frame(anim_args.color_coherence_source, args.outdir, cc_vid_folder, get_frame_name(cc_vid_path), frame_idx, dimensions)

        if turbo_steps == 1 and srt_filename:
            # [source: 47]
            params_string = format_animation_params(keys, prompt_series, effective_frame_idx)
            # [source: 30] # [source: 158]
            write_frame_subtitle(srt_filename, frame_idx, srt_frame_duration, f"F#: {frame_idx}; Cadence: false; Seed: {args.seed}; {params_string}")
            # [source: 48]
            params_string = None # Clear memory

        if frame_idx <= anim_args.max_frames:
            if turbo_steps == 1 or frame_idx == 0:
                print(f"\033[36mFrame in progress:\033[0m {frame_idx} in [0→{anim_args.max_frames}] ")
            else:
                # [source: 49] # [source: 31] # [source: 159]
                print(f"\033[36mFrame in progress:\033[0m {frame_idx} for cadence end range [{frame_idx+1-turbo_steps}→{frame_idx}] in [0→{anim_args.max_frames}] ")

        #_/‾𝓟𝓻𝓮𝓿_𝓲𝓶𝓰‾𝓢𝓮𝓬𝓽𝓲𝓸𝓷‾𝟏‾‾‾‾‾\__________

        if prev_img is not None:
            if anim_args.animation_behavior == 'Normal':
                 # [source: 160]
                 # [source: 50] # [source: 32]-[source: 35]
                 prev_img, depth, matrix_flow, translation_z_estimate = anim_frame_warp(
                     np.copy(prev_img), anim_args, keys, frame_idx,
                     depth_model=depth_model, depth=None, # [source: 161]
                     device=root.device, half_precision=root.half_precision,
                     # [source: 51]
                     inputfiles=inputfiles, hybridframes_path=hybridframes_path, # [source: 162]
                     prev_flow=hybrid_motion_history.get_by_key_or_none(frame_idx-1),
                     prev_img=img_history.get_by_key_or_none(frame_idx-1), # [source: 163]
                     raft_model=raft_model,
                     # [source: 52]
                     z_translation_list=z_translation_history.list_states()
                 )
                 z_translation_history.add_state(translation_z_estimate)

            if anim_args.hybrid_composite == 'Before Motion': # [source: 164]
                # [source: 36]
                prev_img, hybrid_overlay = hybrid_composite(
                    # [source: 53]
                    args, anim_args, frame_idx, np.copy(prev_img),
                    depth_model, hybrid_comp_schedules,
                    inputframes_path, hybridframes_path, root,
                    conform=hybrid_comp_conform, raft_model=raft_model,
                    # [source: 54]
                    return_iterations=return_conform_iterations
                ) # [source: 165]

            if turbo_steps == 1 and anim_args.hybrid_motion_behavior == 'Before Generation' and anim_args.hybrid_motion != 'None':
                # [source: 37]-[source: 38]
                prev_img, prev_motion = hybrid_motion(
                    # [source: 55]
                    frame_idx, np.copy(prev_img),
                    img_history.get_by_key_or_none(frame_idx-1),
                    hybrid_motion_history.get_by_key_or_none(frame_idx-1),
                    args, anim_args, # [source: 166]
                    inputfiles, hybridframes_path, hybrid_flow_factor,
                    # [source: 56]
                    raft_model, matrix_flow=matrix_flow
                )

            if anim_args.hybrid_composite == 'Normal': # [source: 167]
                # [source: 39]
                prev_img, hybrid_overlay = hybrid_composite(
                    # [source: 57]
                    args, anim_args, frame_idx, np.copy(prev_img),
                    depth_model, hybrid_comp_schedules,
                    inputframes_path, hybridframes_path, root,
                    conform=hybrid_comp_conform, raft_model=raft_model,
                    return_iterations=return_conform_iterations
                    # [source: 58]
                ) # [source: 168]

            temporal_history_length = temporal_flow_target_frame + turbo_steps
            if anim_args.temporal_flow != 'None' and temporal_flow_factor != 0 and img_history.length >= temporal_history_length:
                # [source: 40]-[source: 41]
                prev_img = do_temporal_flow(
                    # [source: 59]
                    img_history[temporal_history_length], np.copy(prev_img),
                    anim_args.temporal_flow, frame_idx, raft_model,
                    flow_factor=temporal_flow_factor, # [source: 169]
                    motion_stabilizer_factor=temporal_flow_motion_stabilizer_factor,
                    rotation_stabilizer_factor=temporal_flow_rotation_stabilizer_factor,
                    # [source: 60]
                    return_target=anim_args.temporal_flow_return_target,
                    target_frame=temporal_flow_target_frame,
                    return_flow=False, updown_scale=updown_scale
                ) # [source: 170]

            if anim_args.morpho_flow != 'None' and morpho_iterations != 0:
                # [source: 61]
                prev_img = image_morphological_flow_transform(
                    np.copy(prev_img), anim_args.morpho_image_type,
                    anim_args.morpho_bitmap_threshold, anim_args.morpho_flow,
                    morpho, morpho_iterations, frame_idx, raft_model,
                    flow_factor=morpho_flow_factor, updown_scale=updown_scale, return_flow=False
                    # [source: 62]
                )

            # [source: 171]
            if anim_args.color_coherence != 'None' and anim_args.color_coherence_behavior in ['Before', 'Before/After'] and cc_alpha > 0 and cc_sample is not None and color_matcher is not None:
                # [source: 42]
                prev_img = color_matcher.maintain_colors(
                    # [source: 63]
                    np.copy(prev_img), cc_sample, cc_alpha, anim_args.color_coherence,
                    frame_idx, console_msg="before generation",
                    cc_mix_outdir=cc_mix_outdir, timestring=root.timestring
                )

            if anim_args.color_force_grayscale == 'Before': # [source: 64] # [source: 172]
                prev_img = bgr2gray_bgr(np.copy(prev_img))

            prev_img = (np.copy(prev_img) * contrast).round().astype(np.uint8)

            # [source: 173]
            if amount > 0:
                # [source: 43]
                # Apply mask if use_mask is True and a mask image (from init or video) exists
                # [source: 65]
                apply_mask = mask_image if args.use_mask and mask_image is not None else None
                prev_img = unsharp_mask(np.copy(prev_img), (kernel, kernel), sigma, amount, threshold, apply_mask)

            if args.use_mask or anim_args.use_noise_mask:
                prev_img_for_mask_comp = Image.fromarray(cv2.cvtColor(np.copy(prev_img), cv2.COLOR_BGR2RGB))
                # [source: 66]
                # compose_mask_with_check might modify root.noise_mask
                root.noise_mask = compose_mask_with_check(root, args, noise_mask_seq, noise_mask_vals, prev_img_for_mask_comp)

            # [source: 174]
            prev_img = add_noise(
                np.copy(prev_img), noise, args.seed, anim_args.noise_type,
                # [source: 67]
                (anim_args.perlin_w, anim_args.perlin_h, anim_args.perlin_octaves, anim_args.perlin_persistence),
                root.noise_mask, args.invert_mask
            )

        args.scale = scale

        # [source: 44] # [source: 175]
        if anim_args.animation_mode not in ['Video Input', 'Interpolation']:
             args.strength = min(max(strength, 0.0), 1.0)

        args.pix2pix_img_cfg_scale = float(keys.pix2pix_img_cfg_scale_series[effective_frame_idx])
        # [source: 68]
        # Use effective_frame_idx for prompt and seed schedules
        args.prompt = prompt_series[effective_frame_idx]

        if args.seed_behavior == 'schedule' or use_parseq:
            # [source: 176]
            args.seed = int(keys.seed_schedule_series[effective_frame_idx])

        if anim_args.enable_checkpoint_scheduling:
            args.checkpoint = keys.checkpoint_schedule_series[effective_frame_idx]
            # [source: 69] # [source: 45]
        else:
            args.checkpoint = None

        if anim_args.enable_subseed_scheduling:
            root.subseed = int(keys.subseed_schedule_series[effective_frame_idx])
            # [source: 177]
            root.subseed_strength = float(keys.subseed_strength_schedule_series[effective_frame_idx])

        if use_parseq:
            anim_args.enable_subseed_scheduling = True
            # [source: 70]
            root.subseed = int(keys.subseed_schedule_series[effective_frame_idx])
            # [source: 46]
            root.subseed_strength = keys.subseed_strength_schedule_series[effective_frame_idx]

        args.prompt = prepare_prompt(args.prompt, anim_args.max_frames, args.seed, frame_idx)

        if mode == 'Video Input': # [source: 178]
            args.init_image = get_next_frame(args.outdir, anim_args.video_init_path, frame_idx, False)
            if args.init_image:
                # [source: 71]
                print(f"Using video init frame { args.init_image}")
            else:
                print(f"Warning: Video init frame not found for frame {frame_idx}")
                # Handle missing frame? Maybe stop or use previous? For now, let generate handle it.
                # [source: 72] # [source: 73]
                args.init_image = None # Ensure it's None if not found

        # Re-check video mask inside the loop if mode is Video Input? No, already handled above.
        # [source: 74] # [source: 47] # [source: 179] Redundant check, mask loading moved earlier

        if args.use_mask:
            prev_img_rgb = None if prev_img is None else Image.fromarray(cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB))
            # [source: 180]
            args.mask_image = compose_mask_with_check(root, args, mask_seq, mask_vals, prev_img_rgb) # prev_img_rgb could be None
        else:
            # [source: 75]
            args.mask_image = None # Ensure mask is None if not args.use_mask

        # [source: 48] Use effective_frame_idx for loop schedules
        loop_args.imageStrength = loopSchedulesAndData.image_strength_schedule_series[effective_frame_idx]
        loop_args.blendFactorMax = loopSchedulesAndData.blendFactorMax_series[effective_frame_idx]
        loop_args.blendFactorSlope = loopSchedulesAndData.blendFactorSlope_series[effective_frame_idx]
        loop_args.tweeningFrameSchedule = loopSchedulesAndData.tweening_frames_schedule_series[effective_frame_idx]
        loop_args.colorCorrectionFactor = loopSchedulesAndData.color_correction_factor_series[effective_frame_idx]
        # These don't seem to be schedules, keep as is
        # [source: 76]
        loop_args.use_looper = loopSchedulesAndData.use_looper
        loop_args.imagesToKeyframe = loopSchedulesAndData.imagesToKeyframe


        if 'img2img_fix_steps' in opts.data and opts.data["img2img_fix_steps"]: # [source: 181]
            opts.data["img2img_fix_steps"] = False # Ensure this doesn't interfere unexpectedly
        # [source: 49]
        if scheduled_clipskip is not None:
            opts.data["CLIP_stop_at_last_layers"] = scheduled_clipskip
        if scheduled_noise_multiplier is not None:
            # [source: 77]
            opts.data["initial_noise_multiplier"] = scheduled_noise_multiplier
        if scheduled_ddim_eta is not None:
            opts.data["eta_ddim"] = scheduled_ddim_eta # [source: 182]
        # [source: 50]
        if scheduled_ancestral_eta is not None:
            opts.data["eta_ancestral"] = scheduled_ancestral_eta

        # --- エラー箇所周辺 ---
        # if mode == '3D' and (cmd_opts.lowvram or cmd_opts.medvram): # 修正前
        if mode == '3D' and (getattr(cmd_opts, 'lowvram', False) or getattr(cmd_opts, 'medvram', False)): # 修正後
        # --------------------
            # [source: 78]
            if depth_model and hasattr(depth_model, 'to'): depth_model.to('cpu')
            devices.torch_gc()
            # lowvram.setup_for_low_vram(sd_model, cmd_opts.medvram) # lowvram モジュール/オブジェクトが未定義の可能性
            # sd_hijack.model_hijack.hijack(sd_model) # [source: 183]


        #_/‾𝓟𝓻𝓮𝓿_𝓲𝓶𝓰‾𝓢𝓮𝓬𝓽𝓲𝓸𝓷‾𝟐‾‾‾‾‾\__________

        gen_args = (args, keys, anim_args, loop_args, controlnet_args, root)
        # [source: 51]
        gen_kwargs = {'frame': frame_idx, 'sampler_name': scheduled_sampler_name}

        # [source: 79]
        # --- ControlNetユニット取得と引数設定 ---
        # [source: 265], [source: 278], [source: 52] 検討内容に基づき追加
        controlnet_units = []
        # Check if controlnet_args exists and CN is enabled
        if controlnet_args and is_controlnet_enabled(controlnet_args):
            # get_deforum_controlnet_units に必要な引数を渡す
            # parseq_adapter はループ開始前に初期化済み
            controlnet_units = get_deforum_controlnet_units(
                # [source: 80]
                args, anim_args, controlnet_args, root, parseq_adapter, frame_idx
            ) # [source: 254]
            # ログ出力（任意）
            if controlnet_units:
                 print(f"[Deforum Render] Frame {frame_idx}: Passing {len(controlnet_units)} ControlNet units to generate function.")
            # [source: 81]
            else:
                 # This might happen if units are enabled but loopback frame 0 has no init, or other read errors
                 print(f"[Deforum Render] Frame {frame_idx}: ControlNet is enabled in UI, but no valid units generated for this frame.")


        # generate 関数に controlnet_units をキーワード引数として追加
        # generate 関数側で controlnet_units=None をデフォルト値として受け取れるように修正されている必要あり
        # [source: 82]
        gen_kwargs['controlnet_units'] = controlnet_units
        # ------------------------------------------

        if prev_img is not None: # [source: 184]
            if anim_args.color_force_grayscale in ['Before/After', 'Before']:
                prev_img = bgr2gray_bgr(prev_img)

            seed_store = args.seed # Store seed before potential modification
            # [source: 54]
            args.use_init = True # Set use_init before potentially modifying prev_img
            # [source: 83]

            if int(anim_args.diffusion_redo) > 0: # [source: 185]
                root.init_sample = Image.fromarray(cv2.cvtColor(np.copy(prev_img), cv2.COLOR_BGR2RGB))
                # redo_generation should now accept controlnet_units via **gen_kwargs
                prev_img = redo_generation(np.copy(prev_img), cc_sample, cc_alpha, *gen_args, **gen_kwargs)

            # [source: 84]
            if anim_args.optical_flow_redo_generation != 'None': # [source: 186]
                # [source: 55]
                root.init_sample = Image.fromarray(cv2.cvtColor(np.copy(prev_img), cv2.COLOR_BGR2RGB))
                # optical_flow_generation should now accept controlnet_units via **gen_kwargs
                prev_img = optical_flow_generation(np.copy(prev_img), cc_sample, cc_alpha, redo_flow_factor, raft_model, *gen_args, **gen_kwargs)
                # [source: 85]

            args.seed = seed_store # Restore original seed for main generation if needed
            args.use_init = True # Ensure use_init is still True
            # [source: 187]
            root.init_sample = Image.fromarray(cv2.cvtColor(np.copy(prev_img), cv2.COLOR_BGR2RGB))
        else:
            args.use_init = False # Explicitly set to False if prev_img is None
            # [source: 86]
            root.init_sample = None


        #_/‾𝓡𝓖𝓑‾𝓖𝓮𝓷𝓮𝓻𝓪𝓽𝓲𝓸𝓷‾‾‾‾‾\_______________

        disp.before_generation()
        # [source: 56]
        # --- generate 関数呼び出し (修正済み) ---
        # generate 関数は controlnet_units をキーワード引数として受け取るように修正されている想定
        image = generate(*gen_args, **gen_kwargs) # [source: 188]
        # -------------------------------------
        if image is None:
            # [source: 87]
            print(f"Error: generate function returned None for frame {frame_idx}. Stopping animation.")
            # [source: 88]
            break
        disp.after_generation()

        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # [source: 189] 以降の画像処理 (修正なし)
        # [source: 57]
        if frame_idx == 0 and anim_args.color_coherence != 'None' and anim_args.color_coherence_source in ['Image Path', 'Video Init', 'Video Path'] \
                     and anim_args.color_coherence_behavior in ['Before', 'Before/After'] and cc_sample is not None:
                     # [source: 89]
            # [source: 190]
            # Ensure redo_generation handles **gen_kwargs correctly
            opencv_image = redo_generation(np.copy(opencv_image), cc_sample, cc_alpha, *gen_args, **gen_kwargs, redo_for_cc_before_gen=True)

        if anim_args.hybrid_composite == 'After Generation':
            # [source: 58]
            # [source: 90]
            opencv_image, hybrid_overlay = hybrid_composite(
                args, anim_args, frame_idx, np.copy(opencv_image),
                depth_model, hybrid_comp_schedules,
                inputframes_path, hybridframes_path, root,
                conform=hybrid_comp_conform, raft_model=raft_model,
                return_iterations=return_conform_iterations
                # [source: 91]
            ) # [source: 191]

        if turbo_steps == 1 and anim_args.hybrid_motion_behavior == 'After Generation' and anim_args.hybrid_motion != 'None' and img_history.length >= 1 and frame_idx < anim_args.max_frames:
            # [source: 59]-[source: 60]
            opencv_image, prev_motion = hybrid_motion(
                frame_idx, np.copy(opencv_image),
                img_history.get_by_key_or_none(frame_idx-1),
                # [source: 92]
                hybrid_motion_history.get_by_key_or_none(frame_idx-1), # [source: 192]
                args, anim_args,
                inputfiles, hybridframes_path, hybrid_flow_factor,
                raft_model, matrix_flow=matrix_flow
            )

        if anim_args.animation_behavior == 'After Generation' and turbo_steps == 1: # [source: 93] # [source: 193]
            # [source: 61]-[source: 64]
            opencv_image, depth, matrix_flow, translation_z_estimate = anim_frame_warp(
                np.copy(opencv_image), anim_args, keys, frame_idx,
                depth_model=depth_model, depth=None, # [source: 194]
                device=root.device, half_precision=root.half_precision,
                # [source: 94]
                inputfiles=inputfiles, hybridframes_path=hybridframes_path, # [source: 195]
                prev_flow=hybrid_motion_history.get_by_key_or_none(frame_idx-1), # [source: 196]
                prev_img=img_history.get_by_key_or_none(frame_idx-1), # [source: 197]
                raft_model=raft_model,
                z_translation_list=z_translation_history.list_states()
            )
            # [source: 95]
            z_translation_history.add_state(translation_z_estimate)

        if anim_args.color_coherence != 'None' and color_matcher is not None:
            if cc_sample is None: # [source: 198]
                cc_sample = np.copy(opencv_image)
                # [source: 65]
                print(f"Color match captured from frame {frame_idx}")
            # [source: 96]
            if anim_args.color_coherence_behavior in ['Before/After', 'After'] and cc_alpha > 0 and cc_sample is not None:
                 # [source: 199]
                 opencv_image = color_matcher.maintain_colors(
                     np.copy(opencv_image), cc_sample, cc_alpha, anim_args.color_coherence,
                     # [source: 97]
                     frame_idx, console_msg="after generation",
                     cc_mix_outdir=cc_mix_outdir, timestring=root.timestring
                 )

        if anim_args.color_force_grayscale in ['Before/After', 'After']:
            opencv_image = bgr2gray_bgr(np.copy(opencv_image))

        # [source: 66] # [source: 200]
        if anim_args.hybrid_composite != 'None' and anim_args.hybrid_comp_mask_do_overlay_mask != 'None' and hybrid_overlay is not None:
            # [source: 98]
            overlay_args = (args, anim_args, np.copy(opencv_image), frame_idx, inputframes_path, maskframes_path)
            overlay_kwargs = {'mask_override': hybrid_overlay} if anim_args.hybrid_use_init_image else {'video_mask_override': hybrid_overlay}
            overlay_kwargs.update({'invert_override': anim_args.hybrid_comp_mask_do_overlay_mask != 'Overlay'})
            opencv_image = do_overlay_mask(*overlay_args, **overlay_kwargs)

        # [source: 67] # [source: 201]
        if args.overlay_mask and (anim_args.use_mask_video or args.use_mask):
            # [source: 99]
            opencv_image = do_overlay_mask(args, anim_args, opencv_image.astype(np.uint8), frame_idx, inputframes_path, maskframes_path)

        opencv_image = opencv_image.astype(np.uint8)
        turbo_history.add_state(opencv_image, frame_idx)

        if prev_img is None: # First frame?
            # [source: 100]
            prev_img = np.copy(opencv_image) # [source: 202]

        if turbo_steps == 1:
            # [source: 68] # [source: 203]
            prev_img = make_composite_with_conform(
                loop_comp_type, np.copy(opencv_image), np.copy(prev_img),
                loop_comp_alpha, anim_args.loop_comp_conform_method, raft_model=raft_model,
                conform=loop_comp_conform, order=anim_args.loop_comp_order,
                # [source: 101]
                iterations=anim_args.loop_comp_conform_iterations
            )

        #_/‾𝓒𝓪𝓭𝓮𝓷𝓬𝓮‾‾‾‾‾\_______________________
        # [source: 204] 以降のCadence処理

        turbo_mode = 1 if turbo_steps == 1 else 2 # [source: 209]
        turbo_next_image, next_idx = turbo_history.pop()

        # [source: 69] # [source: 211]
        if anim_args.cadence_save_turbo_frames and outdir_turbo:
            # [source: 102]
            cadence_save(
                'turbo_' + root.timestring, 0 if next_idx == 0 else int(next_idx/turbo_steps),
                outdir_turbo, turbo_next_image,
                anim_args.save_depth_maps, depth_model, depth, anim_args.midas_weight,
                cmd_opts, None, sd_hijack, sd_model, devices, root # lowvram を None に変更 (未定義の可能性があるため)
                # [source: 103]
            )

        if turbo_mode == 1 or next_idx == 0:
             # [source: 70] # [source: 212]
             cadence_save(
                 root.timestring, next_idx, args.outdir, turbo_next_image,
                 anim_args.save_depth_maps, depth_model, depth, anim_args.midas_weight,
                 # [source: 104]
                 cmd_opts, None, sd_hijack, sd_model, devices, root # lowvram を None に変更
             )
             img_history.add_state(turbo_next_image.astype(np.uint8), next_idx)
             hybrid_motion_history.add_state(prev_motion, next_idx)
             disp.add_to_display_queue(bgr_np2pil_rgb(turbo_next_image)) # [source: 213]

        # -------- ここから修正 --------
        if turbo_mode == 2 and next_idx > 0: # [source: 105] Cadenceブロック開始
            last_next_idx = next_idx - turbo_steps
            turbo_prev_image = img_history.get_by_key(last_next_idx) # [source: 215]
            prev_idx = last_next_idx + 1
            start_idx = prev_idx
            # [source: 216]
            end_idx = start_idx + turbo_steps # [source: 106]

            # [source: 72]
            print(f"\033[94mCadence setup for tween between frames:\033[0m {start_idx} to {end_idx-1} ")

            depth = None # Reset depth for cadence loop
            # --- 新しい変数名を使用 ---
            # build_turbo_dict は新しい辞書を生成して返すと仮定
            cadence_turbo_data = build_turbo_dict( # [source: 107]
                {}, # 空の辞書を渡す (または `turbo` を渡す必要がある場合: `turbo,`)
                turbo_prev_image, turbo_next_image, start_idx, end_idx, keys,
                img_history.get_by_key_or_none(last_next_idx),
                hybrid_motion_history.get_by_key_or_none(last_next_idx)
            )
            # -------------------------

            # --- ↓↓↓ ここに移動したコード (新しい変数名で) ↓↓↓ ---
            # 前回のサイクルからの 'last_turbo' キーがあれば削除
            if 'last_turbo' in cadence_turbo_data: # [source: 217] のロジックをここに移動
                del cadence_turbo_data['last_turbo']
            # --- ↑↑↑ 移動完了 ↑↑↑ ---

            # --- 以降の処理では新しい変数名を使用 ---
            cadence_turbo_data, translation_z_estimates = cadence_animation_frames( # [source: 108]
                cadence_turbo_data, depth, depth_model, inputfiles, hybridframes_path,
                anim_args, keys, root, z_translation_list=z_translation_history.list_states()
            )
            # [source: 73] # [source: 218]
            for item in translation_z_estimates:
                z_translation_history.add_state(item)

            if anim_args.hybrid_composite == 'Before Motion' and anim_args.hybrid_comp_conform_method != 'None':
                # [source: 109]
                cadence_turbo_data = cadence_hybrid_comp(cadence_turbo_data, inputframes_path, dimensions, args, anim_args, raft_model)

            # [source: 219]
            if anim_args.hybrid_motion != 'None':
                # [source: 74]
                cadence_turbo_data = cadence_hybrid_motion(cadence_turbo_data, inputfiles, inputframes_path, hybridframes_path, args, anim_args, raft_model)
                # [source: 110]

            if anim_args.hybrid_composite in ['Normal', 'After Generation'] and anim_args.hybrid_comp_conform_method != 'None':
                cadence_turbo_data = cadence_hybrid_comp(cadence_turbo_data, inputframes_path, dimensions, args, anim_args, raft_model)

            # [source: 220]
            cadence_turbo_data = cadence_temporal_flow(cadence_turbo_data, anim_args, raft_model)
            cadence_turbo_data = cadence_morpho(cadence_turbo_data, anim_args, raft_model)
            # [source: 111] # [source: 75] # [source: 221]
            cadence_turbo_data = cadence_flow(cadence_turbo_data, anim_args.optical_flow_cadence, raft_model, updown_scale)

            # _/‾𝓒𝓪𝓭𝓮𝓷𝓬𝓮‾𝓛𝓸𝓸𝓹‾‾‾‾‾\_________________

            for cadence_idx in range(start_idx, end_idx):
                # Check interruption at the start of each cadence frame processing
                if state.interrupted:
                    # [source: 112]
                    print("\n** Interrupted by user during cadence loop **")
                    break # Exit cadence loop

                state.job = f"frame {cadence_idx + 1}/{anim_args.max_frames+1}"
                # [source: 222]
                # [source: 113]
                state.job_no = cadence_idx + 1
                # [source: 76]
                if state.skipped:
                    print("\n** PAUSED **")
                    state.skipped = False
                    # [source: 114]
                    while not state.skipped:
                         time.sleep(0.1) # [source: 223]
                         if state.interrupted: # Allow interruption while paused
                             print("\n** Interrupted by user while paused in cadence **")
                             # [source: 115]
                             break # Exit inner loop
                    if state.interrupted:
                        break # Exit outer cadence loop
                    # [source: 116] # [source: 77]
                    print("** RESUMING **")

                # Ensure cadence_idx is valid key for cadence_turbo_data dict
                if cadence_idx not in cadence_turbo_data:
                    print(f"Warning: Cadence index {cadence_idx} not found in cadence_turbo_data dictionary. Skipping.")
                    # [source: 117]
                    continue

                # [source: 224]
                t = cadence_turbo_data[cadence_idx]
                turbo_prev, turbo_next, depth, ckey = t['prev'], t['next'], t['depth'], t['ckey']
                # [source: 118]
                tween = ckey['tween']

                if srt_filename:
                    # [source: 78]
                     params_string = format_animation_params(keys, prompt_series, cadence_idx) # Use cadence_idx here
                     # [source: 79] # [source: 226]
                     # [source: 119]
                     write_frame_subtitle(srt_filename, cadence_idx, srt_frame_duration, f"F#: {cadence_idx}; Cadence: {tween['prime'] < 1.0}; Seed: {args.seed}; {params_string}")
                     params_string = None # Clear memory

                if tween['prime'] < 1.0:
                     img = turbo_prev * (1.0 - tween['image']) + turbo_next * tween['image'] # [source: 227]
                     # [source: 120]
                # [source: 80]
                else:
                    img = turbo_next

                # [source: 228]
                # [source: 121]
                img = img.astype(np.uint8)

                if anim_args.color_coherence != 'None' and color_matcher is not None:
                    # [source: 81] # [source: 229]
                    if anim_args.color_coherence_in_cadence and anim_args.color_coherence_behavior not in ['Before'] and anim_args.color_coherence_source in ['Video Init', 'Video Path'] and ckey['cc_alpha'] != 0 and cc_vid_folder is not None:
                        # [source: 122] # [source: 230]
                        cc_sample_cadence = get_cc_sample_from_video_frame(anim_args.color_coherence_source, args.outdir, cc_vid_folder, get_frame_name(cc_vid_path), cadence_idx, dimensions)
                        # Use cc_sample_cadence if available, otherwise fallback to original cc_sample
                        # [source: 123]
                        current_cc_sample = cc_sample_cadence if cc_sample_cadence is not None else cc_sample
                    else:
                        current_cc_sample = cc_sample

                    if current_cc_sample is not None:
                        # [source: 124]
                        img = color_matcher.maintain_colors(
                            img, current_cc_sample, ckey['cc_alpha'], anim_args.color_coherence,
                            cadence_idx, suppress_console=True, console_msg="in cadence after generation",
                            # [source: 125]
                            cc_mix_outdir=cc_mix_outdir, timestring=root.timestring
                        )

                # [source: 82] # [source: 231]
                if anim_args.color_force_grayscale in ['Before/After', 'After']:
                    img = bgr2gray_bgr(img)

                # [source: 126] # [source: 232]
                if anim_args.hybrid_composite != 'None' and anim_args.hybrid_comp_mask_do_overlay_mask != 'None' and hybrid_overlay is not None:
                    # [source: 83]
                    overlay_args = (args, anim_args, img, cadence_idx, inputframes_path, maskframes_path)
                    # [source: 127]
                    overlay_kwargs = {'mask_override': hybrid_overlay} if anim_args.hybrid_use_init_image else {'video_mask_override': hybrid_overlay}
                    overlay_kwargs.update({'invert_override': anim_args.hybrid_comp_mask_do_overlay_mask != 'Overlay'})
                    # [source: 233]
                    img = do_overlay_mask(*overlay_args, **overlay_kwargs)

                # [source: 128] # [source: 84]
                if args.overlay_mask and (anim_args.use_mask_video or args.use_mask):
                     img = do_overlay_mask(args, anim_args, img, cadence_idx, inputframes_path, maskframes_path)

                # [source: 234]
                cadence_save(
                    # [source: 129]
                    root.timestring, cadence_idx, args.outdir, img,
                    anim_args.save_depth_maps, depth_model, depth, anim_args.midas_weight,
                    cmd_opts, None, sd_hijack, sd_model, devices, root # lowvram を None に変更
                )
                img_history.add_state(img.astype(np.uint8), cadence_idx)
                # [source: 130]
                # Use the motion from the *start* of the cadence sequence for all frames within it
                # Or should this be interpolated/updated? Assuming fixed motion for now.
                # [source: 131]
                hybrid_motion_history.add_state(prev_motion, cadence_idx)
                # [source: 85] # [source: 236]
                disp.add_to_display_queue(bgr_np2pil_rgb(img))

                if cadence_idx == end_idx-1:
                    turbo_prev_image, prev_idx = np.copy(img), end_idx

                # [source: 132] # [source: 237] # [source: 86]
                prev_img = make_composite_with_conform(
                    ckey['loop_comp_type'], np.copy(img), np.copy(prev_img),
                    ckey['loop_comp_alpha'], anim_args.loop_comp_conform_method, raft_model=raft_model,
                    # [source: 133]
                    conform=ckey['loop_comp_conform'], order=anim_args.loop_comp_order,
                    iterations=anim_args.loop_comp_conform_iterations
                )

                # [source: 238]
                print_cadence_msg(cadence_idx, anim_args.optical_flow_cadence, ckey, tween, anim_args.max_frames)

            # Check if cadence loop was interrupted before proceeding
            # [source: 134]
            if state.interrupted:
                break # Exit main loop if interrupted during cadence

            # Clean up cadence-specific variables using the new name
            del(turbo_prev, turbo_next, ckey, tween, cadence_turbo_data)
            gc.collect()
        # -------- ここまで修正 --------

        if frame_idx <= anim_args.max_frames: # [source: 239]
            # [source: 87]
            print(f"\033[94mFrame complete:\033[0m {frame_idx}")

        # Check interruption before incrementing frame index
        if state.interrupted:
             print("\n** Interruption detected after frame processing. Stopping loop. **")
             # [source: 136]
             break

        frame_idx += turbo_steps
        args.seed = next_seed(args, root)

    #_/‾𝓒𝓵𝓮𝓪𝓷𝓾𝓹‾‾‾‾‾\________________________
    # [source: 240] からの後処理
    if depth_model and hasattr(depth_model, 'delete_model') and not opts.data.get("deforum_keep_3d_models_in_vram", False):
        depth_model.delete_model()
        del depth_model
        devices.torch_gc()

    if raft_model and hasattr(raft_model, 'delete_model'):
        # [source: 137]
        raft_model.delete_model()
        del raft_model
        devices.torch_gc()

    disp.stop()
    opts.live_previews_enable = live_previews_enable_store # Restore original setting

    print("\n-- Animation Render Finished --")

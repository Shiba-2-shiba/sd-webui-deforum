# This helper script is responsible for ControlNet/Deforum integration
# Modified for sd-webui-forge environment

import os
import copy
import gradio as gr
import numpy as np
from PIL import Image
import lib_controlnet # For error handling check

# --- lib_controlnetからのインポート ---
from lib_controlnet.global_state import (
    update_controlnet_filenames,
    get_all_preprocessor_names,
    get_all_controlnet_names,
    get_sorted_preprocessors,
    get_preprocessor
)
from lib_controlnet.external_code import (
    ControlNetUnit, ControlMode, ResizeMode,
    resize_mode_from_value, control_mode_from_value
)
from lib_controlnet.utils import load_state_dict # 必要なら
from lib_controlnet.logging import logger # 必要なら
from lib_controlnet.controlnet_ui.tool_button import ToolButton # Use Forge's ToolButton

# --- Deforum (reforum) 固有のインポート (既存または修正済み) ---
# ToolButtonのローカル定義は不要になるため、deforum_controlnet_gradio.pyから削除されている想定
from .deforum_controlnet_gradio import hide_ui_by_cn_status, hide_file_textboxes
from .general_utils import count_files_in_folder, clean_gradio_path_strings, debug_print # Forgeでの動作確認
from .video_audio_utilities import SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_VIDEO_EXTENSIONS, get_extension_if_valid, vid2frames, convert_image # Forgeでの動作確認
from .animation_key_frames import ControlNetKeys # Forgeでの動作確認
from .load_images import load_image # Forgeでの動作確認

# --- その他必要な標準ライブラリ (既存) ---
from modules import shared, scripts

# number of CN model tabs to show in the deforum gui.
# Respect A1111 setting 'control_net_max_models_num', but ensure at least 5 tabs if the setting is lower.
max_models = shared.opts.data.get("control_net_max_models_num", 5) # Default to 5 if not set
num_of_models = max(5, max_models) # Ensure at least 5 models

def controlnet_infotext():
    # Kept original infotext, assuming links are still relevant or informative
    # sd-webui-forge 内の ControlNet 統合を使用することを示すように調整
    return """<p>Requires the built-in ControlNet integration within sd-webui-forge.</p>
           <p>Ensure you have downloaded ControlNet models and placed them in the correct directory specified in Forge's settings.</p>
           """

def is_controlnet_enabled(controlnet_args):
    """Checks if any ControlNet unit is enabled in the provided arguments."""
    if not controlnet_args:
        return False
    for i in range(1, num_of_models + 1):
        # getattrを使用して安全にアクセス
        if getattr(controlnet_args, f'cn_{i}_enabled', False):
            return True
    return False

# ==========================================================================================
# Function to get ControlNetUnit list for the current frame
# This function replaces the old process_with_controlnet from A1111 versions
# ==========================================================================================
def get_deforum_controlnet_units(args, anim_args, controlnet_args, root, parseq_adapter, frame_idx) -> list[ControlNetUnit]:
    """
    Generates a list of ControlNetUnit objects for the current frame based on Deforum settings.

    Args:
        args: General Deforum arguments.
        anim_args: Animation specific arguments.
        controlnet_args: ControlNet specific arguments from the UI.
        root: Root object containing state like init_sample.
        parseq_adapter: Adapter for Parseq keyframes (if used).
        frame_idx: The current frame index.

    Returns:
        A list of configured ControlNetUnit objects for the current frame.
    """
    units = []
    if not is_controlnet_enabled(controlnet_args):
        return units

    # Initialize ControlNetKeys based on whether Parseq is used
    # Assuming parseq_adapter has a 'use_parseq' attribute and potentially 'cn_keys'
    if parseq_adapter and hasattr(parseq_adapter, 'use_parseq') and parseq_adapter.use_parseq and hasattr(parseq_adapter, 'cn_keys'):
        CnSchKeys = parseq_adapter.cn_keys # Use keys from Parseq adapter if available
        debug_print("Using Parseq ControlNet keys.")
    else:
        # Create keys directly from args if Parseq is not used or adapter lacks keys
        CnSchKeys = ControlNetKeys(anim_args, controlnet_args)
        debug_print("Using standard Deforum ControlNet keys.")

    def read_cn_data(cn_idx: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Reads ControlNet input image and mask numpy arrays for a given unit index and frame."""
        cn_mask_np, cn_image_np = None, None
        prefix = f'cn_{cn_idx}'

        # === Loopback Mode Logic ===
        if getattr(controlnet_args, f'{prefix}_loopback_mode', False):
            debug_print(f"ControlNet Unit {cn_idx}: Loopback mode enabled for frame {frame_idx}.")
            # Frame 0: Use init image if available
            if frame_idx == 0:
                init_img_source = None
                if args.use_init:
                    # Prioritize init_image_box if it exists and has content
                    init_img_box_val = getattr(args, 'init_image_box', None)
                    if isinstance(init_img_box_val, dict) and init_img_box_val.get('image') is not None:
                         init_img_source = init_img_box_val['image']
                         debug_print(f"ControlNet Unit {cn_idx}: Using init_image_box for frame 0 loopback.")
                    elif args.init_image:
                         init_img_source = args.init_image
                         debug_print(f"ControlNet Unit {cn_idx}: Using init_image path for frame 0 loopback.")

                if init_img_source:
                    try:
                        # load_image should return PIL image or None
                        pil_image = load_image(init_img_source)
                        if pil_image:
                            cn_image_np = np.array(pil_image.convert("RGB")).astype('uint8')
                            debug_print(f"ControlNet Unit {cn_idx}: Loaded init image successfully for frame 0 loopback.")
                        else:
                            debug_print(f"ControlNet Unit {cn_idx}: load_image returned None for frame 0 loopback source.")
                    except Exception as e:
                        logger.error(f"Error loading init image '{init_img_source}' for ControlNet Unit {cn_idx} frame 0 loopback: {e}")
                else:
                     debug_print(f"ControlNet Unit {cn_idx}: No suitable init image found for frame 0 loopback.")
            # Subsequent frames: Use previous frame's output (root.init_sample)
            elif frame_idx > 0:
                if root.init_sample is not None:
                    # Ensure root.init_sample is a PIL Image before converting
                    if isinstance(root.init_sample, Image.Image):
                        cn_image_np = np.array(root.init_sample.convert("RGB")).astype('uint8')
                        debug_print(f"ControlNet Unit {cn_idx}: Using root.init_sample (PIL) for frame {frame_idx} loopback.")
                    elif isinstance(root.init_sample, np.ndarray): # Handle case if it's already numpy
                        # Assuming BGR needs conversion to RGB if necessary, or direct use if already RGB
                        # Let's assume it's already in the correct format expected downstream or convert
                        cn_image_np = root.init_sample.astype('uint8')
                        debug_print(f"ControlNet Unit {cn_idx}: Using root.init_sample (numpy) for frame {frame_idx} loopback.")
                    else:
                         debug_print(f"ControlNet Unit {cn_idx}: root.init_sample is not PIL or numpy array for frame {frame_idx} loopback.")
                else:
                    debug_print(f"ControlNet Unit {cn_idx}: root.init_sample is None for frame {frame_idx} loopback.")

        # === Non-Loopback Mode (File Input) Logic ===
        else:
            # --- Load Base Image ---
            cn_inputframes_path = os.path.join(args.outdir, f'controlnet_{cn_idx}_inputframes')
            if os.path.isdir(cn_inputframes_path):
                num_files = count_files_in_folder(cn_inputframes_path)
                if num_files > 0:
                    # Use first frame if only one exists (static image), else use current frame index
                    # Ensure file extensions are handled (assuming .jpg based on original)
                    frame_file = "000000000.jpg" if num_files == 1 else f"{frame_idx:09}.jpg"
                    cn_frame_path = os.path.join(cn_inputframes_path, frame_file)
                    if os.path.exists(cn_frame_path):
                        try:
                            pil_image = Image.open(cn_frame_path).convert("RGB")
                            cn_image_np = np.array(pil_image).astype('uint8')
                            if num_files == 1:
                                debug_print(f'ControlNet Unit {cn_idx}: Reading *static* base frame from {cn_frame_path}')
                            else:
                                debug_print(f'ControlNet Unit {cn_idx}: Reading base frame #{frame_idx} from {cn_frame_path}')
                        except Exception as e:
                            logger.error(f"Error loading ControlNet base frame {cn_frame_path}: {e}")
                    else:
                        debug_print(f"ControlNet Unit {cn_idx}: Base frame file not found: {cn_frame_path}")
                else:
                    debug_print(f"ControlNet Unit {cn_idx}: Base input directory is empty: {cn_inputframes_path}")
            else:
                 debug_print(f"ControlNet Unit {cn_idx}: Base input directory not found: {cn_inputframes_path}")

            # --- Load Mask Image (if path exists) ---
            cn_maskframes_path = os.path.join(args.outdir, f'controlnet_{cn_idx}_maskframes')
            if os.path.isdir(cn_maskframes_path):
                num_files = count_files_in_folder(cn_maskframes_path)
                if num_files > 0:
                    mask_file = "000000000.jpg" if num_files == 1 else f"{frame_idx:09}.jpg"
                    cn_mask_frame_path = os.path.join(cn_maskframes_path, mask_file)
                    if os.path.exists(cn_mask_frame_path):
                        try:
                            # Forge ControlNet expects single channel mask (0-255)
                            # Load as grayscale and convert to numpy
                            pil_mask = Image.open(cn_mask_frame_path).convert("L")
                            cn_mask_np = np.array(pil_mask).astype('uint8')
                            # Ensure it has the channel dimension expected by Forge (H, W, 1) or (H, W) - check Forge needs
                            # Assuming Forge might handle (H, W) directly or (H, W, 1). Let's keep it (H, W) first.
                            # cn_mask_np = cn_mask_np[:, :, np.newaxis] # Uncomment if (H, W, 1) is needed

                            if num_files == 1:
                                debug_print(f'ControlNet Unit {cn_idx}: Reading *static* mask frame from {cn_mask_frame_path}')
                            else:
                                debug_print(f'ControlNet Unit {cn_idx}: Reading mask frame #{frame_idx} from {cn_mask_frame_path}')
                        except Exception as e:
                            logger.error(f"Error loading ControlNet mask frame {cn_mask_frame_path}: {e}")
                    else:
                        debug_print(f"ControlNet Unit {cn_idx}: Mask frame file not found: {cn_mask_frame_path}")
                # else: Mask directory exists but is empty - no mask applied
            # else: Mask directory doesn't exist - no mask applied

        return cn_mask_np, cn_image_np

    # --- Main logic for get_deforum_controlnet_units ---
    for i in range(1, num_of_models + 1):
        prefix = f"cn_{i}"
        enabled = getattr(controlnet_args, f"{prefix}_enabled", False)

        # Skip if not enabled in UI
        if not enabled:
            continue

        # Read image/mask data for this unit
        mask_np, image_np = read_cn_data(i)

        # Further checks: Disable unit if loopback is on, it's frame 0, and read_cn_data failed to get an image
        if getattr(controlnet_args, f"{prefix}_loopback_mode", False) and frame_idx == 0 and image_np is None:
            enabled = False
            debug_print(f"ControlNet Unit {i}: Disabling for frame 0 loopback due to missing init image.")

        # Skip if disabled after loopback check or if no input image is available (CN requires an image)
        if not enabled:
            continue
        if image_np is None:
            logger.warning(f"[Deforum Warning] ControlNet Unit {i} is enabled but no input image could be found or loaded for frame {frame_idx}. Skipping.")
            continue

        # Prepare image dictionary for ControlNetUnit
        # Create a zero mask if no mask is provided but an image is
        if mask_np is None:
             # Forge expects single channel mask (H, W) or (H, W, 1)
             mask_np = np.zeros_like(image_np[:, :, 0], dtype=np.uint8) # Create single channel zero mask (H, W)
             # If Forge needs (H, W, 1):
             # mask_np = mask_np[:, :, np.newaxis]
        else:
             # Ensure loaded mask is single channel if it wasn't already
             if mask_np.ndim == 3 and mask_np.shape[2] > 1:
                 mask_np = mask_np[:, :, 0] # Take the first channel, assuming grayscale was loaded as RGB

        # Ensure mask dimensions match image dimensions
        if mask_np.shape[0] != image_np.shape[0] or mask_np.shape[1] != image_np.shape[1]:
             logger.warning(f"ControlNet Unit {i}: Mask dimensions ({mask_np.shape}) do not match image dimensions ({image_np.shape[:2]}). Resizing mask.")
             mask_np = cv2.resize(mask_np, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
             # Ensure correct dimensions after resize (H, W)
             if mask_np.ndim == 3: mask_np = mask_np[:,:,0]


        image_dict = {'image': image_np, 'mask': mask_np}

        # Get scheduled values safely, providing defaults
        # Ensure series have enough length for frame_idx
        def get_scheduled_value(key_name, default_value):
            series = getattr(CnSchKeys, f"{key_name}_schedule_series", [default_value])
            # Handle cases where series might be shorter than expected
            if frame_idx < len(series):
                return series[frame_idx]
            elif series: # Not empty, return last value
                return series[-1]
            else: # Empty series, return default
                return default_value

        weight = float(get_scheduled_value(f"{prefix}_weight", 1.0))
        guidance_start = float(get_scheduled_value(f"{prefix}_guidance_start", 0.0))
        guidance_end = float(get_scheduled_value(f"{prefix}_guidance_end", 1.0))

        # Get other parameters from controlnet_args
        module = getattr(controlnet_args, f"{prefix}_module", "none")
        model = getattr(controlnet_args, f"{prefix}_model", "None")
        resize_mode_str = getattr(controlnet_args, f"{prefix}_resize_mode", "Crop and Resize") # Default to Inner Fit
        processor_res = int(getattr(controlnet_args, f"{prefix}_processor_res", 512))
        threshold_a = float(getattr(controlnet_args, f"{prefix}_threshold_a", -1)) # Use defaults from preprocessor if -1
        threshold_b = float(getattr(controlnet_args, f"{prefix}_threshold_b", -1)) # Use defaults from preprocessor if -1
        pixel_perfect = getattr(controlnet_args, f"{prefix}_pixel_perfect", False)
        control_mode_str = getattr(controlnet_args, f"{prefix}_control_mode", "Balanced")
        # low_vram = getattr(controlnet_args, f"{prefix}_low_vram", False) # Check if ControlNetUnit in Forge uses this

        # Create ControlNetUnit object
        try:
            unit = ControlNetUnit(
                enabled=True, # Already checked enabled status
                module=module,
                model=model,
                weight=weight,
                image=image_dict,
                resize_mode=resize_mode_from_value(resize_mode_str), # Convert string to Enum
                # low_vram=low_vram, # Include if needed by Forge's CNUnit
                processor_res=processor_res,
                threshold_a=threshold_a,
                threshold_b=threshold_b,
                guidance_start=guidance_start,
                guidance_end=guidance_end,
                pixel_perfect=pixel_perfect,
                control_mode=control_mode_from_value(control_mode_str), # Convert string to Enum
                # hr_option: Forge's CN script handles this based on its own UI/settings
            )
            units.append(unit)
            debug_print(f"ControlNet Unit {i}: Added. Mod={unit.module}, Mdl={unit.model}, W={unit.weight:.2f}, GS={unit.guidance_start:.2f}, GE={unit.guidance_end:.2f}, PP={unit.pixel_perfect}, CM={unit.control_mode.value}")
        except Exception as e:
            logger.error(f"Error creating ControlNetUnit {i} for frame {frame_idx}: {e}")
            logger.error(f"  Parameters: module={module}, model={model}, weight={weight}, resize_mode={resize_mode_str}, control_mode={control_mode_str}")

    if units:
         debug_print(f"Generated {len(units)} ControlNetUnit(s) for frame {frame_idx}.")
    else:
         debug_print(f"No active ControlNetUnit(s) generated for frame {frame_idx}.")

    return units

# ==========================================================================================
# Setup ControlNet UI elements using Forge's API
# ==========================================================================================
def setup_controlnet_ui_raw():
    """Sets up the raw Gradio UI elements for Deforum's ControlNet integration."""
    # Ensure ControlNet model/module lists are up-to-date
    try:
        update_controlnet_filenames()
        cn_models = get_all_controlnet_names()
        # Use sorted preprocessors for better UI order
        sorted_preprocessors_dict = get_sorted_preprocessors()
        # Get names list and prepend "none"
        preprocessor_names_list = ["none"] + list(sorted_preprocessors_dict.keys())
    except Exception as e:
        logger.error(f"Failed to get ControlNet models or preprocessors: {e}")
        cn_models = ["None"]
        preprocessor_names_list = ["none"]


    # Helper function to get Gradio update kwargs for sliders based on Preprocessor info
    def get_slider_update(preprocessor, slider_name, is_pixel_perfect):
        """Gets update arguments for a slider based on preprocessor capabilities."""
        slider_info = getattr(preprocessor, slider_name, None)
        # Check if the preprocessor object and the specific slider attribute exist
        if slider_info and hasattr(slider_info, 'gradio_update_kwargs'):
             kwargs = slider_info.gradio_update_kwargs.copy()
             # Hide resolution slider if pixel perfect is enabled
             if slider_name == 'slider_resolution' and is_pixel_perfect:
                  kwargs['visible'] = False
                  kwargs['interactive'] = False
             # Ensure interactive is False if not visible
             if not kwargs.get('visible', False):
                  kwargs['interactive'] = False
             return kwargs
        # Return default hidden state if slider info not found or invalid
        return {'visible': False, 'interactive': False, 'label': slider_name.replace('slider_', '').capitalize(), 'value': -1, 'minimum': -1, 'maximum': -1, 'step': 1}

    # Function to generate Gradio updates for a selected preprocessor
    def build_sliders(module: str, pp_is_checked: bool):
        """Builds Gradio update list for sliders and visibility based on selected preprocessor.""" # <-- この行からインデント開始 (通常スペース4つ)
        preprocessor = None # デフォルトは None
        if module != "none": # "none" 以外の場合のみ取得を試みる
            try:
                preprocessor = get_preprocessor(module) # Get Preprocessor object
            except KeyError:
             # ForgeのControlNetに 'none' がないためのエラーと想定
                 logger.warning(f"Preprocessor '{module}' not found or 'none' requested. ControlNet sliders will be based on 'none'.")
             # preprocessor は None のままにする

    # --- スライダー処理 ---
    # get_slider_update が preprocessor=None を安全に扱えると仮定 (source: 258-261)
        res_kwargs = get_slider_update(preprocessor, 'slider_resolution', pp_is_checked)
        slider1_kwargs = get_slider_update(preprocessor, 'slider_1', pp_is_checked)
        slider2_kwargs = get_slider_update(preprocessor, 'slider_2', pp_is_checked)

    # --- 表示状態決定 --- (preprocessorがNoneの場合も考慮)
        advanced_visible = res_kwargs.get('visible', False) or \
                           slider1_kwargs.get('visible', False) or \
                           slider2_kwargs.get('visible', False)
        model_needed = False
        control_mode_needed = False
        if preprocessor is not None: # preprocessor オブジェクトが取得できた場合のみチェック
            model_needed = not preprocessor.do_not_need_model
            control_mode_needed = preprocessor.show_control_mode

    # return 文も含め、関数本体の最後まで同じインデントレベルを維持
        return [
            gr.update(**res_kwargs),
            gr.update(**slider1_kwargs),
            gr.update(**slider2_kwargs),
            gr.update(visible=advanced_visible),
            gr.update(visible=model_needed), # モデル表示
            gr.update(visible=model_needed), # 更新ボタン表示
            gr.update(visible=control_mode_needed) # ControlMode表示
        ]

    # Callback to refresh ControlNet models
    def refresh_all_models(current_selection):
        """Refreshes the ControlNet model dropdown choices."""
        try:
            update_controlnet_filenames()
            all_cn_models = get_all_controlnet_names()
            # Keep current selection if it still exists, otherwise default to "None"
            selected = current_selection if current_selection in all_cn_models else "None"
            return gr.Dropdown.update(value=selected, choices=["None"] + all_cn_models)
        except Exception as e:
            logger.error(f"Error refreshing ControlNet models: {e}")
            return gr.Dropdown.update(choices=["None"], value="None") # Fallback

    # --- UI Definition ---
    model_dropdowns = [] # Keep track of model dropdowns if needed elsewhere
    infotext_fields = [] # For infotext generation (if used by Deforum)

    def create_model_in_tab_ui(cn_id: int) -> dict:
        """Creates the UI elements for a single ControlNet tab."""
        with gr.Row():
            enabled = gr.Checkbox(label="Enable", value=False, elem_id=f"deforum_cn_{cn_id}_enabled")
            pixel_perfect = gr.Checkbox(label="Pixel Perfect", value=False, elem_id=f"deforum_cn_{cn_id}_pixel_perfect", visible=True, interactive=True)
            # Low VRAM Checkbox (Visibility depends on global setting in Forge? Assume visible for now)
            low_vram = gr.Checkbox(label="Low VRAM", value=False, elem_id=f"deforum_cn_{cn_id}_low_vram", visible=True, interactive=True) # Might not be used by Forge CNUnit directly
            overwrite_frames = gr.Checkbox(label='Overwrite input frames', value=True, elem_id=f"deforum_cn_{cn_id}_overwrite_frames", visible=False, interactive=True) # Renamed key
            loopback_mode = gr.Checkbox(label="LoopBack mode", value=False, elem_id=f"deforum_cn_{cn_id}_loopback_mode", visible=False, interactive=True) # Renamed key

        # Rows for module, model, weight, guidance (initially hidden)
        with gr.Row(visible=False) as mod_row:
            module = gr.Dropdown(preprocessor_names_list, label=f"Preprocessor", value="none", elem_id=f"deforum_cn_{cn_id}_module")
            model = gr.Dropdown(["None"] + cn_models, label=f"Model", value="None", elem_id=f"deforum_cn_{cn_id}_model")
            refresh_models = ToolButton(value='\U0001f504') # Refresh symbol
            refresh_models.click(refresh_all_models, inputs=[model], outputs=[model], show_progress=False) # Refresh this dropdown

        with gr.Row(visible=False) as weight_row:
            weight = gr.Textbox(label="Weight schedule", lines=1, value='0:(1.0)', elem_id=f"deforum_cn_{cn_id}_weight")

        with gr.Row(visible=False) as guidance_row: # Combine start/end for better UI space
            guidance_start = gr.Textbox(label="Guidance Start schedule", lines=1, value='0:(0.0)', elem_id=f"deforum_cn_{cn_id}_guidance_start")
            guidance_end = gr.Textbox(label="Guidance End schedule", lines=1, value='0:(1.0)', elem_id=f"deforum_cn_{cn_id}_guidance_end")

        # Advanced options accordion (initially hidden)
        with gr.Accordion("Advanced options", open=False, visible=False) as advanced_column:
             # Use -1 as default/placeholder, actual defaults depend on selected preprocessor
            processor_res = gr.Slider(label="Preprocessor Resolution", minimum=64, maximum=2048, value=512, step=64, interactive=True, elem_id=f"deforum_cn_{cn_id}_processor_res") # Renamed key
            threshold_a = gr.Slider(label="Threshold A", minimum=-1, maximum=255, value=-1, step=1, interactive=True, elem_id=f"deforum_cn_{cn_id}_threshold_a") # Renamed key
            threshold_b = gr.Slider(label="Threshold B", minimum=-1, maximum=255, value=-1, step=1, interactive=True, elem_id=f"deforum_cn_{cn_id}_threshold_b") # Renamed key

        # File input row (initially hidden, toggled by loopback)
        with gr.Row(visible=True) as file_input_row: # Start visible, hide if loopback is checked
             vid_path = gr.Textbox(value='', label="Input Video/Image Path", elem_id=f"deforum_cn_{cn_id}_vid_path")
             mask_vid_path = gr.Textbox(value='', label="Mask Video/Image Path", elem_id=f"deforum_cn_{cn_id}_mask_vid_path")

        # Mode rows (initially hidden)
        with gr.Row(visible=False) as mode_row:
            resize_mode = gr.Radio(
                 choices=[e.value for e in ResizeMode], # Use Enum values
                 value=ResizeMode.INNER_FIT.value, # Default Enum value ('Crop and Resize')
                 label="Resize Mode", elem_id=f"deforum_cn_{cn_id}_resize_mode")

        with gr.Row(visible=False) as control_mode_row:
            control_mode = gr.Radio(
                choices=[e.value for e in ControlMode], # Use Enum values
                value=ControlMode.BALANCED.value, # Default Enum value
                label="Control Mode", elem_id=f"deforum_cn_{cn_id}_control_mode")


        # --- UI Event Handling ---
        components_to_show = [
            pixel_perfect, low_vram, mod_row, weight_row, guidance_row,
            advanced_column, # File input row is handled separately by loopback
            mode_row, control_mode_row,
            overwrite_frames, loopback_mode # Checkboxes also need visibility toggled
        ]
        # Show/hide main components based on the "Enable" checkbox
        for component in components_to_show:
             enabled.change(fn=hide_ui_by_cn_status, inputs=[enabled], outputs=[component], show_progress=False)
        # Also toggle file input visibility based on enable
        enabled.change(fn=lambda is_enabled: gr.update(visible=is_enabled and not loopback_mode.value), inputs=[enabled], outputs=[file_input_row], show_progress=False)


        # Show/hide file input textboxes based on the "LoopBack mode" checkbox
        # Only relevant if the unit is enabled
        loopback_mode.change(
             fn=lambda is_loopback, is_enabled: gr.update(visible=is_enabled and not is_loopback),
             inputs=[loopback_mode, enabled],
             outputs=[file_input_row],
             show_progress=False
        )

        # Update sliders and dependent component visibility when preprocessor changes
        module.change(
            fn=build_sliders,
            inputs=[module, pixel_perfect],
            outputs=[processor_res, threshold_a, threshold_b, advanced_column, model, refresh_models, control_mode_row],
            show_progress=False
        )
        # Update sliders when pixel perfect changes
        pixel_perfect.change(
            fn=build_sliders,
            inputs=[module, pixel_perfect],
            outputs=[processor_res, threshold_a, threshold_b, advanced_column, model, refresh_models, control_mode_row],
            show_progress=False
        )

        # Store model dropdown if needed elsewhere
        model_dropdowns.append(model)

        # Store fields for infotext generation (use keys consistent with ControlNetUnit)
        infotext_fields.extend([
            (enabled, f"ControlNet {cn_id} Enabled"), # Needs special handling if not in ControlNetUnit
            (module, f"ControlNet {cn_id} Preprocessor"),
            (model, f"ControlNet {cn_id} Model"),
            (weight, f"ControlNet {cn_id} Weight"), # Schedule needs special handling for infotext
            (guidance_start, f"ControlNet {cn_id} Guidance Start"), # Schedule needs special handling
            (guidance_end, f"ControlNet {cn_id} Guidance End"), # Schedule needs special handling
            (resize_mode, f"ControlNet {cn_id} Resize Mode"),
            (control_mode, f"ControlNet {cn_id} Control Mode"),
            (pixel_perfect, f"ControlNet {cn_id} Pixel Perfect"),
            (processor_res, f"ControlNet {cn_id} Preprocessor Resolution"),
            (threshold_a, f"ControlNet {cn_id} Threshold A"),
            (threshold_b, f"ControlNet {cn_id} Threshold B"),
            (vid_path, f"ControlNet {cn_id} Input Path"), # Not part of ControlNetUnit, handled separately
            (mask_vid_path, f"ControlNet {cn_id} Mask Path"), # Not part of ControlNetUnit
            (loopback_mode, f"ControlNet {cn_id} Loopback Mode"), # Not part of ControlNetUnit
            (overwrite_frames, f"ControlNet {cn_id} Overwrite Frames"), # Not part of ControlNetUnit
            (low_vram, f"ControlNet {cn_id} Low VRAM"), # Not part of ControlNetUnit
            # hr_option is not in this UI, Forge CN handles it
        ])

        # Return dictionary of Gradio components for this tab
        # Keys should match the names expected by Deforum when collecting args
        return {
            "enabled": enabled, "pixel_perfect": pixel_perfect, "low_vram": low_vram,
            "module": module, "model": model, "weight": weight,
            "guidance_start": guidance_start, "guidance_end": guidance_end,
            "processor_res": processor_res, "threshold_a": threshold_a, "threshold_b": threshold_b,
            "resize_mode": resize_mode, "control_mode": control_mode,
            "overwrite_frames": overwrite_frames, "vid_path": vid_path,
            "mask_vid_path": mask_vid_path, "loopback_mode": loopback_mode
        }

    # --- Main UI Setup ---
    # Use gr.Blocks context to ensure structure, though we primarily return a dict
    with gr.Blocks() as controlnet_ui_root:
        with gr.Accordion('🕹️ ControlNet Integration (sd-webui-forge)', open=False): # Accordion instead of TabItem
            gr.HTML(controlnet_infotext()) # Show info text
            with gr.Tabs():
                model_params = {}
                # Create UI tabs for the specified number of models
                for i in range(1, num_of_models + 1):
                    with gr.Tab(f"ControlNet Unit {i}"):
                        # Create the UI elements and store them in the dictionary
                        model_params[i] = create_model_in_tab_ui(i)

                # Flatten the parameters into a single dictionary for Gradio interface
                flat_params = {}
                for i in range(1, num_of_models + 1):
                    for key, value in model_params[i].items():
                        # Create the key name expected by Deforum args (e.g., cn_1_enabled)
                        flat_params[f"cn_{i}_{key}"] = value

    # Return the flattened dictionary of all UI components
    return flat_params


# ==========================================================================================
# Function to setup the ControlNet UI, handling potential errors
# ==========================================================================================
def setup_controlnet_ui():
    """Sets up the Deforum ControlNet UI for sd-webui-forge, with error handling."""
    try:
        # Basic check if lib_controlnet seems available
        # Check for a key component like external_code or global_state
        if not hasattr(lib_controlnet, 'external_code') or not hasattr(lib_controlnet, 'global_state'):
             raise ImportError("Key components of lib_controlnet not found. Is sd-webui-forge ControlNet properly installed/integrated?")

        logger.info("Setting up Deforum ControlNet UI for Forge...")
        ui_components = setup_controlnet_ui_raw()
        logger.info("Deforum ControlNet UI setup complete.")
        return ui_components
    except Exception as e:
        logger.error(f"Error setting up Deforum ControlNet UI for Forge: {e}", exc_info=True)
        # Provide a Gradio HTML component indicating failure
        return {'controlnet_error_placeholder': gr.HTML(f"""
                <div style='padding: 10px; border: 1px solid red; background-color: #fee; margin: 10px;'>
                    <strong>Failed to initialize Deforum ControlNet UI for Forge.</strong><br>
                    Reason: {e}<br>
                    Please ensure sd-webui-forge and its ControlNet components are correctly installed and functional.
                    Check the console log (enable --controlnet-loglevel debug for more details) for more details. ControlNet functionality will be disabled.
                </div>
                """, visible=True)}

# ==========================================================================================
# Function to return the names of all ControlNet components for Gradio interface definition
# ==========================================================================================
def controlnet_component_names() -> list[str]:
    """Returns a list of all Gradio component names generated by setup_controlnet_ui."""
    names = []
    # Define the suffixes for components within each ControlNet unit
    # Must match the keys in the dictionary returned by create_model_in_tab_ui
    component_suffixes = [
        "enabled", "pixel_perfect", "low_vram", "module", "model", "weight",
        "guidance_start", "guidance_end", "processor_res", "threshold_a", "threshold_b",
        "resize_mode", "control_mode", "overwrite_frames", "vid_path",
        "mask_vid_path", "loopback_mode"
    ]
    # Generate names for each unit
    for i in range(1, num_of_models + 1):
        for suffix in component_suffixes:
            names.append(f'cn_{i}_{suffix}')

    # Add the placeholder key in case of error
    #names.append('controlnet_error_placeholder')
    return names

# ==========================================================================================
# Video Unpacking Logic (Mostly unchanged, verify paths and helpers)
# ==========================================================================================

def process_controlnet_input_frames(args, anim_args, controlnet_args, video_path, mask_path, outdir_suffix, unit_idx):
    """Processes a single video/image input for a CN unit, extracting frames."""
    prefix = f'cn_{unit_idx}'
    # --- Add Debug Print ---
    print(f"--- [DEBUG CN {unit_idx}] ---")
    print(f"Function called with video_path: {video_path}, mask_path: {mask_path}")
    ui_vid_path = getattr(controlnet_args, f'{prefix}_vid_path', 'N/A')
    ui_mask_path = getattr(controlnet_args, f'{prefix}_mask_vid_path', 'N/A')
    print(f"Value from UI textbox (vid_path): {ui_vid_path}")
    print(f"Value from UI textbox (mask_vid_path): {ui_mask_path}")
    # ----------------------
    if not getattr(controlnet_args, f'{prefix}_enabled', False):
        return # Skip if the unit is not enabled
    if getattr(controlnet_args, f'{prefix}_loopback_mode', False):
        debug_print(f"ControlNet Unit {unit_idx}: Skipping frame extraction due to Loopback mode.")
        return # Skip if loopback mode is enabled

    is_mask_processing = mask_path is not None
    target_path = mask_path if is_mask_processing else video_path
    # --- Add More Debug Print ---
    print(f"Target path for processing: {target_path}")
    if target_path:
        target_path_cleaned = clean_gradio_path_strings(target_path) # Assuming clean_gradio_path_strings exists and works
        print(f"Cleaned target path: {target_path_cleaned}")
        exists = os.path.exists(target_path_cleaned)
        print(f"Path exists: {exists}")
        if not exists:
            logger.error(f"ControlNet Unit {unit_idx}: {input_type.capitalize()} input path does not exist after cleaning: {target_path_cleaned}")
            return
    else:
        print("Target path is empty, skipping.")
        return
     # ... rest of the function
    input_type = 'mask' if is_mask_processing else 'base'

    if not target_path:
        # This case should ideally not happen if loopback is false and enabled is true,
        # but check just in case UI logic allows it.
        logger.warning(f"ControlNet Unit {unit_idx}: Enabled but no {input_type} video/image path provided (and not in loopback mode). Skipping extraction.")
        return

    target_path = clean_gradio_path_strings(target_path)
    if not target_path or not os.path.exists(target_path):
         logger.error(f"ControlNet Unit {unit_idx}: {input_type.capitalize()} input path does not exist: {target_path}")
         return

    frame_path = os.path.join(args.outdir, f'controlnet_{unit_idx}_{outdir_suffix}')
    overwrite = getattr(controlnet_args, f'{prefix}_overwrite_frames', True)

    # Check if overwrite is False and directory already has content
    if not overwrite and os.path.isdir(frame_path) and any(os.scandir(frame_path)):
         debug_print(f"ControlNet Unit {unit_idx}: Skipping {input_type} frame extraction as directory exists and overwrite is disabled: {frame_path}")
         return

    os.makedirs(frame_path, exist_ok=True)

    is_image = get_extension_if_valid(target_path, SUPPORTED_IMAGE_EXTENSIONS)
    is_video = get_extension_if_valid(target_path, SUPPORTED_VIDEO_EXTENSIONS)

    if is_image:
        # Copy single image to the frame directory, named as the first frame (000...0.jpg)
        try:
            dest_file = os.path.join(frame_path, '000000000.jpg')
            # Use reforum's convert_image helper which should handle format conversion
            convert_image(target_path, dest_file)
            logger.info(f"ControlNet Unit {unit_idx}: Copied {input_type} image to: {dest_file}")
        except Exception as e:
             logger.error(f"Error copying/converting ControlNet Unit {unit_idx} {input_type} image '{target_path}': {e}")
    elif is_video:
        # Extract frames from video
        logger.info(f'Unpacking ControlNet Unit {unit_idx} {input_type} video: {target_path} to {frame_path}')
        try:
            # Determine frame extraction range based on animation mode (same logic as original)
            extract_nth = 1 if anim_args.animation_mode != 'Video Input' else anim_args.extract_nth_frame
            extract_from = 0 if anim_args.animation_mode != 'Video Input' else anim_args.extract_from_frame
            extract_to = anim_args.max_frames if anim_args.animation_mode != 'Video Input' else anim_args.extract_to_frame # Check vid2frames logic for inclusivity

            # Use reforum's vid2frames helper
            vid2frames(
                video_path=target_path,
                video_in_frame_path=frame_path,
                n=extract_nth,
                overwrite=overwrite, # Use the value from checkbox
                extract_from_frame=extract_from,
                extract_to_frame=extract_to, # Ensure vid2frames handles this correctly
                numeric_files_output=True,
                out_ext='jpg' # Ensure output is jpg for consistency
            )
            logger.info(f"ControlNet Unit {unit_idx} {input_type} video unpacked successfully.")
        except Exception as e:
            logger.error(f"Error unpacking ControlNet Unit {unit_idx} {input_type} video '{target_path}': {e}")
    else:
        logger.error(f"ControlNet Unit {unit_idx}: Input path '{target_path}' for {input_type} is not a supported video or image format.")


def unpack_controlnet_vids(args, anim_args, controlnet_args):
    """Called once per run to extract frames for all active non-loopback ControlNet units."""
    logger.info("Unpacking ControlNet input videos/images (if any)...")
    if not controlnet_args:
         logger.info("ControlNet arguments not provided, skipping unpack.")
         return

    for i in range(1, num_of_models + 1):
        prefix = f'cn_{i}'
        if not getattr(controlnet_args, f'{prefix}_enabled', False):
            continue # Skip disabled units

        # Don't extract if loopback is enabled
        if getattr(controlnet_args, f'{prefix}_loopback_mode', False):
            continue

        # Get paths for base and mask
        vid_path = getattr(controlnet_args, f'{prefix}_vid_path', None)
        mask_path = getattr(controlnet_args, f'{prefix}_mask_vid_path', None)

        # Process base video/image if path is provided
        if vid_path:
            process_controlnet_input_frames(args, anim_args, controlnet_args, vid_path, None, 'inputframes', i)

        # Process mask video/image if path is provided
        if mask_path:
            process_controlnet_input_frames(args, anim_args, controlnet_args, None, mask_path, 'maskframes', i)

    logger.info("ControlNet input unpacking finished.")
    # Original returned vid_paths_used, but it doesn't seem to be used elsewhere. Returning None for now.
    return None

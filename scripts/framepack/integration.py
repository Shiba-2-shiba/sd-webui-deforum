# integration.py (LoRA/FP8対応 最終版)

import os
import torch
import gc
import sys
import traceback

# eichiから採用する機能と、新しいManagerに対応するための機能をインポート
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

from . import tensor_tool
from .memory import (
    offload_model_from_device_for_memory_preservation,
    move_model_to_device_with_memory_preservation,
    get_cuda_free_memory_gb,
)
from .transformer_manager import TransformerManager
from .text_encoder_manager import TextEncoderManager
from .vae_manager import VaeManager
from .discovery import FramepackDiscovery
# FP8警告フラグのリセット関数をインポート
from .lora_utils.fp8_optimization_utils import reset_warning_flags


class ImageEncoderManager:
    """Image Encoder (CLIP Vision)のロードとライフサイクルを管理するクラス"""
    def __init__(self, device, model_path: str):
        self.model = None
        self.device = device
        self.is_loaded = False
        if not model_path or not os.path.isdir(model_path):
            raise FileNotFoundError(f"ImageEncoderManager received an invalid model_path: {model_path}")
        self.model_path = model_path

    def get_model(self):
        if not self.is_loaded:
            print(f"Loading Image Encoder from: {self.model_path}")
            from transformers import CLIPVisionModelWithProjection
            self.model = CLIPVisionModelWithProjection.from_pretrained(
                self.model_path, subfolder="image_encoder", torch_dtype=torch.bfloat16,
                local_files_only=True, ignore_mismatched_sizes=True
            ).cpu()
            self.model.eval()
            self.is_loaded = True
        return self.model

    def dispose(self):
        if self.model is not None:
            print("Disposing Image Encoder...")
            del self.model
            self.model = None
            self.is_loaded = False


class ImageProcessorManager:
    """Image Processorのロードとライフサイクルを管理するクラス"""
    def __init__(self, model_path: str):
        self.processor = None
        self.is_loaded = False
        if not model_path or not os.path.isdir(model_path):
            raise FileNotFoundError(f"ImageProcessorManager received an invalid model_path: {model_path}")
        self.model_path = model_path

    def get_processor(self):
        if not self.is_loaded:
            print(f"Loading Image Processor from: {self.model_path}")
            from transformers import SiglipImageProcessor
            self.processor = SiglipImageProcessor.from_pretrained(
                self.model_path, subfolder="feature_extractor", local_files_only=True
            )
            self.is_loaded = True
        return self.processor

    def dispose(self):
        if self.processor is not None:
            print("Disposing Image Processor...")
            del self.processor
            self.processor = None
            self.is_loaded = False


class TokenizerManager:
    """Tokenizerのロードとライフサイクルを管理するクラス"""
    def __init__(self, model_path: str):
        self.tokenizer = None
        self.tokenizer_2 = None
        self.is_loaded = False
        if not model_path or not os.path.isdir(model_path):
            raise FileNotFoundError(f"TokenizerManager received an invalid model_path: {model_path}")
        self.model_path = model_path

    def get_tokenizers(self):
        if not self.is_loaded:
            print(f"Loading Tokenizers from: {self.model_path}")
            from transformers import LlamaTokenizerFast, CLIPTokenizer
            self.tokenizer = LlamaTokenizerFast.from_pretrained(self.model_path, subfolder="tokenizer", local_files_only=True)
            self.tokenizer_2 = CLIPTokenizer.from_pretrained(self.model_path, subfolder="tokenizer_2", local_files_only=True)
            self.is_loaded = True
        return self.tokenizer, self.tokenizer_2

    def dispose(self):
        if self.tokenizer is not None:
            print("Disposing Tokenizers...")
            del self.tokenizer; del self.tokenizer_2
            self.tokenizer = None; self.tokenizer_2 = None
            self.is_loaded = False


class FramepackIntegration:
    """FramePack F1のモデル管理とビデオ生成を統合する司令塔クラス。"""
    def __init__(self, device):
        self.device = device
        self.managers = None
        self.sdxl_components = None
        self.discovery = FramepackDiscovery()
        self.last_used_f1_args = {} # 完了時アラームのために引数を保持

    def _initialize_managers(self, local_paths: dict[str, str]):
        global_managers = {}
        free_mem_gb = get_cuda_free_memory_gb(self.device)
        high_vram = free_mem_gb > 16

        print("Initializing managers with explicit model paths...")
        
        # ★★★ 修正箇所: 新しいTransformerManagerの呼び出し ★★★
        global_managers["transformer"] = TransformerManager(
            device=self.device, 
            high_vram_mode=high_vram, 
            model_path=local_paths.get("transformer") # F1モデルのパスを渡す
        )
        global_managers["text_encoder"] = TextEncoderManager(
            device=self.device, 
            high_vram_mode=high_vram,
            model_path=local_paths.get("text_encoder")
        )
        global_managers["image_encoder"] = ImageEncoderManager(
            device=self.device, 
            model_path=local_paths.get("flux_bfl")
        )
        global_managers["image_processor"] = ImageProcessorManager(
            model_path=local_paths.get("flux_bfl")
        )
        global_managers["vae"] = VaeManager(
            device=self.device, 
            high_vram_mode=high_vram,
            model_path=local_paths.get("vae")
        )
        global_managers["tokenizers"] = TokenizerManager(
            model_path=local_paths.get("text_encoder")
        )
        
        print("All managers initialized.")
        return global_managers

    def _get_sdxl_components(self, sd_model):
        from modules.sd_models import FakeInitialModel
        if isinstance(sd_model, FakeInitialModel):
            return {"unet": None, "vae": None, "text_encoders": None}
            
        components = {"unet": None, "vae": None, "text_encoders": None}
        if hasattr(sd_model, "model") and hasattr(sd_model.model, "diffusion_model"):
            components["unet"] = sd_model.model.diffusion_model
        elif hasattr(sd_model, "unet"):
            components["unet"] = sd_model.unet
        if hasattr(sd_model, "first_stage_model"):
            components["vae"] = sd_model.first_stage_model
        if hasattr(sd_model, "cond_stage_model"):
            components["text_encoders"] = sd_model.cond_stage_model
        return components

    def setup_environment(self, sd_model):
        print("Step 1: Checking for required local models...")
        models_exist, missing_repos = self.discovery.check_models_exist()

        if not models_exist:
            error_message = (
                "One or more FramePack F1 models are missing. Please download them manually before running Deforum.\n"
                "Missing repositories:\n"
            )
            for repo in missing_repos: error_message += f" - {repo}\n"
            error_message += "You can use the standalone downloader script if needed: 'python extensions/sd-forge-deforum/scripts/framepack/model_downloader.py'"
            raise FileNotFoundError(error_message)
        
        print("All required models found locally.")

        print("Step 2: Resolving local paths...")
        local_paths = {
            "transformer": self.discovery.get_local_path("transformer"),
            "text_encoder": self.discovery.get_local_path("text_encoder"),
            "vae": self.discovery.get_local_path("vae"),
            "flux_bfl": self.discovery.get_local_path("flux_bfl"),
        }
        
        for name, path in local_paths.items():
            if path is None or not os.path.isdir(path):
                raise RuntimeError(f"Failed to resolve a valid directory path for component '{name}': {path}")
            print(f"  - Resolved {name}: {path}")

        print("Step 3: Initializing model managers with resolved paths...")
        self.managers = self._initialize_managers(local_paths)

        print("Step 4: Offloading base SD model from VRAM...")
        self.sdxl_components = self._get_sdxl_components(sd_model)
        if self.sdxl_components["unet"]: offload_model_from_device_for_memory_preservation(self.sdxl_components["unet"], self.device)
        if self.sdxl_components["vae"]: offload_model_from_device_for_memory_preservation(self.sdxl_components["vae"], self.device)
        if self.sdxl_components["text_encoders"]: offload_model_from_device_for_memory_preservation(self.sdxl_components["text_encoders"], self.device)
        
        gc.collect()
        torch.cuda.empty_cache()
        print("Environment setup complete.")

    def generate_video(self, args, anim_args, video_args, framepack_f1_args, root):
        """
        動画生成処理を外部モジュール `tensor_tool.py` に委譲します。
        LoRA/FP8の設定をTransformerManagerに渡し、モデルの状態を更新します。
        """
        print("[FramePack Integration] Starting video generation process...")
        self.last_used_f1_args = framepack_f1_args # 完了時アラームのために引数を保存

        try:
            # ★★★ 1. FP8警告フラグのリセット (eichi採用機能) ★★★
            reset_warning_flags()
            print("Reset FP8 warning flags for new generation.")

            # ★★★ 2. LoRA/FP8設定をTransformerManagerに適用 ★★★
            transformer_manager = self.managers["transformer"]
            high_vram_mode = transformer_manager.current_state['high_vram']

            # UIからの引数を安全に取得
            lora_paths = getattr(framepack_f1_args, 'lora_paths', [])
            lora_scales = getattr(framepack_f1_args, 'lora_scales', [])
            fp8_enabled = getattr(framepack_f1_args, 'fp8_enabled', True) # デフォルトで有効化

            # 新しい設定をTransformerManagerにセット
            transformer_manager.set_next_settings(
                lora_paths=lora_paths,
                lora_scales=lora_scales,
                fp8_enabled=fp8_enabled,
                high_vram_mode=high_vram_mode
            )
            
            # 設定変更に基づき、モデルをリロード（必要に応じて）
            transformer_manager.ensure_transformer_state()
            
            # 他のマネージャーの状態も確認
            self.managers["text_encoder"].ensure_text_encoder_state()

            # ★★★ 3. コア生成ロジックの実行 ★★★
            print("[FramePack Integration] Delegating to tensor_tool module...")
            returned_images = tensor_tool.execute_generation(
                managers=self.managers,
                device=self.device,
                args=args,
                anim_args=anim_args,
                video_args=video_args,
                framepack_f1_args=framepack_f1_args,
                root=root
            )

            if returned_images and isinstance(returned_images, list) and len(returned_images) > 0:
                print(f"[FramePack Integration] Generation completed successfully. {len(returned_images)} images returned.")
            else:
                print("[FramePack Integration] Generation finished, but no images were returned from tensor_tool.")

            return returned_images

        except Exception as e:
            print(f"[FramePack Integration] An error occurred during video generation.")
            traceback.print_exc()
            raise e

    def cleanup_environment(self):
        """環境をクリーンアップし、eichi採用の完了時アラーム機能を実行します。"""
        if self.managers is None:
            print("Cleanup skipped: managers were not initialized.")
            return

        print("Cleaning up FramePack F1 environment...")
        
        try:
            for manager_name, manager in self.managers.items():
                if manager and hasattr(manager, 'dispose'):
                    print(f"Disposing manager: {manager_name}...")
                    manager.dispose()
            
            self.managers = None
            gc.collect(); torch.cuda.empty_cache()

            print("Restoring base SD model to VRAM...")
            if self.sdxl_components:
                if self.sdxl_components["unet"]: move_model_to_device_for_memory_preservation(self.sdxl_components["unet"], self.device)
                if self.sdxl_components["vae"]: move_model_to_device_for_memory_preservation(self.sdxl_components["vae"], self.device)
                if self.sdxl_components["text_encoders"]: move_model_to_device_for_memory_preservation(self.sdxl_components["text_encoders"], self.device)
            
            print("Cleanup complete.")

        finally:
            # ★★★ 完了時アラーム機能 (eichi採用機能) ★★★
            play_alarm = getattr(self.last_used_f1_args, 'alarm_on_completion', False)
            if play_alarm:
                print("Playing completion sound...")
                # Windowsでのみサウンドを再生
                if HAS_WINSOUND and sys.platform == 'win32':
                    try:
                        winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
                    except Exception as alarm_error:
                        print(f"Failed to play completion sound: {alarm_error}")
                else:
                    # 他のOS向けの代替通知（コンソール出力）
                    print("\n\a======================\n    PROCESSING COMPLETE\n======================\a\n")

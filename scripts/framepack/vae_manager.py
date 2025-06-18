import os
import torch
from ..diffusers import AutoencoderKLHunyuanVideo
from .memory import DynamicSwapInstaller

class VaeManager:
    # --- 修正箇所 1: __init__ に model_path を追加 ---
    def __init__(self, device, high_vram_mode: bool, model_path: str):
        self.vae = None
        self.device = device
        self.high_vram_mode = high_vram_mode
        self.is_loaded = False

        # 受け取ったパスを検証し、保存する
        if not model_path or not os.path.isdir(model_path):
            raise FileNotFoundError(f"VaeManager received an invalid model_path: {model_path}")
        self.model_path = model_path

    # --- 修正箇所 2: _load_model で self.model_path を使用 ---
    def _load_model(self):
        print(f"Loading Hunyuan VAE from: {self.model_path}")
        # ハードコードされたIDの代わりに、保存したパスを使用し、ローカルファイルのみを指定
        self.vae = AutoencoderKLHunyuanVideo.from_pretrained(
            self.model_path,
            subfolder='vae',
            torch_dtype=torch.bfloat16,
            local_files_only=True
        ).cpu()
        self.vae.eval()
        self.vae.requires_grad_(False)

        if self.high_vram_mode:
            self.vae.to(self.device)
        else:
            self.vae.enable_slicing()
            self.vae.enable_tiling()
            DynamicSwapInstaller.install_model(self.vae, device=self.device)

        self.is_loaded = True
        print("Hunyuan VAE loaded.")

    def get_model(self):
        if not self.is_loaded:
            self._load_model()
        return self.vae

    def dispose(self):
        if self.vae is not None:
            # 既存の dispose ロジックは変更不要
            del self.vae
            self.vae = None
            self.is_loaded = False
            torch.cuda.empty_cache()
            print("Hunyuan VAE disposed.")

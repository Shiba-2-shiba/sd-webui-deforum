import os
import glob
import torch
import traceback
import gc
from accelerate import init_empty_weights

# 必要なコンポーネントをインポート
from .hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from .memory import DynamicSwapInstaller, cpu, get_cuda_free_memory_gb

# lora_utilsから必要な関数をインポート
# lora_utilsフォルダが同じ階層にあることを確認してください。
from .lora_utils.lora_loader import load_and_apply_lora
from .lora_utils.fp8_optimization_utils import apply_fp8_monkey_patch, check_fp8_support

class TransformerManager:
    """
    Transformerモデルの状態管理を行うクラス（LoRAおよびFP8対応版）
    
    このクラスは以下の責務を持ちます：
    - Transformerモデルのライフサイクル管理
    - LoRA設定の管理と適用
    - FP8最適化の管理と適用
    
    設定の変更はすぐには適用されず、次回のリロード時に適用されます。
    """

    def __init__(self, device, high_vram_mode: bool, model_path: str):
        self.transformer = None
        self.device = device

        if not model_path or not os.path.exists(model_path):
            # HuggingFace HubのIDである可能性も考慮し、パス存在チェックはリロード時に行う
            print(f"TransformerManager initialized with model path or ID: {model_path}")
        self.model_path = model_path

        # 現在適用されている設定
        self.current_state = {
            'lora_paths': [],
            'lora_scales': [],
            'fp8_enabled': False,
            'is_loaded': False,
            'high_vram': high_vram_mode
        }

        # 次回のロード時に適用する設定
        self.next_state = self.current_state.copy()

        # 最初に空のモデルを仮想デバイスにロードしておく
        self._load_virtual_transformer()
        print("Virtual Transformer model loaded.")
        
    def set_next_settings(self, lora_paths=None, lora_scales=None, fp8_enabled=True, high_vram_mode=False):
        """
        次回のロード時に使用する設定をセットします。
        FP8はデフォルトで有効化されます。
        """
        if lora_paths is None: lora_paths = []
        if lora_scales is None: lora_scales = []

        if not isinstance(lora_paths, list): lora_paths = [lora_paths]
        if not isinstance(lora_scales, list): lora_scales = [lora_scales]
        
        if lora_paths and not lora_scales:
            lora_scales = [0.8] * len(lora_paths)
        
        if len(lora_scales) < len(lora_paths):
            lora_scales.extend([0.8] * (len(lora_paths) - len(lora_scales)))
        elif len(lora_scales) > len(lora_paths):
            lora_scales = lora_scales[:len(lora_paths)]

        self.next_state = {
            'lora_paths': lora_paths,
            'lora_scales': lora_scales,
            'fp8_enabled': fp8_enabled,
            'high_vram': high_vram_mode,
            'is_loaded': self.current_state['is_loaded'],
        }
    
    def _needs_reload(self):
        """現在の状態と次回の設定を比較し、リロードが必要かどうかを判断します。"""
        if not self._is_loaded():
            return True

        # LoRAのパスまたはスケールの変更をチェック
        if set(self.current_state['lora_paths']) != set(self.next_state['lora_paths']):
            return True
        if self.current_state['lora_scales'] != self.next_state['lora_scales']:
            return True
        
        # FP8最適化設定の変更をチェック
        if self.current_state['fp8_enabled'] != self.next_state['fp8_enabled']:
            return True

        # High-VRAMモードの変更をチェック
        if self.current_state['high_vram'] != self.next_state['high_vram']:
            return True

        return False
    
    def _is_loaded(self):
        """Transformerがロードされているか確認します。"""
        return self.transformer is not None and self.current_state['is_loaded']
    
    def get_transformer(self):
        """現在のTransformerインスタンスを取得します。"""
        return self.transformer

    def ensure_transformer_state(self):
        """Transformerの状態を確認し、必要に応じてリロードします。"""
        if self._needs_reload():
            print("Transformer settings have changed. Reloading model...")
            return self._reload_transformer()        
        print("Using pre-loaded transformer model.")
        return True
    
    def _load_virtual_transformer(self):
        """仮想デバイスに空のTransformerモデルをロードします。"""
        try:
            with init_empty_weights():
                # from_pretrainedはローカルパスもHub IDも受け付ける
                config = HunyuanVideoTransformer3DModelPacked.load_config(self.model_path)
                self.transformer = HunyuanVideoTransformer3DModelPacked.from_config(config, torch_dtype=torch.bfloat16)
            self.transformer.to(torch.bfloat16)
        except Exception as e:
            raise RuntimeError(f"Failed to load virtual transformer config from '{self.model_path}'. Ensure the path is correct. Error: {e}")

    def _find_model_files(self):
        """指定されたモデルパスから状態辞書のファイルを取得します。"""
        if os.path.isdir(self.model_path):
            # ローカルパスの場合
            model_files = glob.glob(os.path.join(self.model_path, '**', '*.safetensors'), recursive=True)
            if not model_files:
                 model_files = glob.glob(os.path.join(self.model_path, '**', '*.bin'), recursive=True)
            model_files.sort()
            return model_files
        else:
            # Hugging Face Hub IDの場合
            from huggingface_hub import snapshot_download
            # snapshot_downloadはキャッシュへのパスを返す
            model_root = snapshot_download(repo_id=self.model_path)
            model_files = glob.glob(os.path.join(model_root, '**', '*.safetensors'), recursive=True)
            if not model_files:
                model_files = glob.glob(os.path.join(model_root, '**', '*.bin'), recursive=True)
            model_files.sort()
            return model_files

    def _reload_transformer(self):
        """next_stateの設定でTransformerをリロードします。"""
        try:
            self.dispose() # 既存のモデルを解放

            print("Reloading Transformer with new settings...")
            print(f"  - LoRA Paths: {[os.path.basename(p) for p in self.next_state['lora_paths']]}")
            print(f"  - LoRA Scales: {self.next_state['lora_scales']}")
            print(f"  - FP8 Enabled: {self.next_state['fp8_enabled']}")
            print(f"  - High VRAM Mode: {self.next_state['high_vram']}")

            lora_paths = self.next_state['lora_paths']
            fp8_enabled = self.next_state['fp8_enabled']

            # LoRAまたはFP8が有効な場合、state_dictベースでロード
            if lora_paths or fp8_enabled:
                if fp8_enabled:
                    has_e4m3, _, _ = check_fp8_support()
                    if not has_e4m3:
                        print("FP8 is enabled but not supported by the current PyTorch environment. Disabling FP8.")
                        self.next_state['fp8_enabled'] = False
                        fp8_enabled = False

                model_files = self._find_model_files()
                if not model_files:
                    raise FileNotFoundError(f"Could not find model files for '{self.model_path}'.")

                state_dict = load_and_apply_lora(
                    model_files, 
                    lora_paths,
                    self.next_state['lora_scales'],
                    fp8_enabled,
                    device=self.device
                )

                # 仮想モデルに重みをロード
                self._load_virtual_transformer()
                
                if fp8_enabled:
                    print("Applying FP8 monkey patch to the model...")
                    apply_fp8_monkey_patch(self.transformer, state_dict, use_scaled_mm=False)

                print("Loading state_dict into the model...")
                self.transformer.load_state_dict(state_dict, assign=True, strict=True)
                
                # メモリ解放
                del state_dict
                gc.collect()

            # LoRAもFP8も無効な場合、from_pretrainedでシンプルにロード
            else:
                self.transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16
                )

            # 共通のセットアップ処理
            self.transformer.cpu().eval()
            self.transformer.high_quality_fp32_output_for_inference = True
            self.transformer.requires_grad_(False)
            
            # TeaCacheをデフォルトで有効化
            print("Enabling TeaCache by default...")
            self.transformer.initialize_teacache()
            
            # VRAMモードに応じたデバイス配置
            if self.next_state['high_vram']:
                print("High VRAM mode: Moving entire transformer to GPU...")
                self.transformer.to(self.device)
            else:
                print("Low VRAM mode: Applying DynamicSwapInstaller...")
                DynamicSwapInstaller.install_model(self.transformer, device=self.device)
            
            self.next_state['is_loaded'] = True
            self.current_state = self.next_state.copy()
            
            print("Transformer reload complete.")
            return True
            
        except Exception as e:
            print(f"Transformer reload failed: {e}")
            traceback.print_exc()
            self.current_state['is_loaded'] = False
            return False

    def dispose(self):
        """Transformerモデルのリソースを解放します。"""
        if self.transformer is None:
            return

        print("Disposing Transformer model...")
        
        # is_loaded状態はリロードの成否に依存するため、high_vramモードはcurrent_stateから直接確認
        if not self.current_state.get('high_vram', True):
            print("Uninstalling DynamicSwapInstaller patches from Transformer...")
            DynamicSwapInstaller.uninstall_model(self.transformer)
        
        try:
            self.transformer.to(cpu)
        except Exception as e:
            print(f"Could not move transformer to cpu: {e}")

        del self.transformer
        self.transformer = None
        self.current_state['is_loaded'] = False
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Transformer disposed.")

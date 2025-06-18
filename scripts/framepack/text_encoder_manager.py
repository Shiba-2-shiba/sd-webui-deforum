import os
import torch
import traceback
import gc
from .memory import DynamicSwapInstaller

class TextEncoderManager:
    """text_encoderとtext_encoder_2の状態管理を行うクラス
    
    このクラスは以下の責務を持ちます：
    - text_encoderとtext_encoder_2のライフサイクル管理
    
    設定の変更はすぐには適用されず、次回のリロード時に適用されます。
    """

    def __init__(self, device, high_vram_mode: bool, model_path: str):
        self.text_encoder = None
        self.text_encoder_2 = None
        self.device = device

        if not model_path or not os.path.isdir(model_path):
            raise FileNotFoundError(f"TextEncoderManager received an invalid model_path: {model_path}")
        self.model_path = model_path

        self.current_state = {
            'is_loaded': False,
            'high_vram': high_vram_mode
        }

        self.next_state = self.current_state.copy()
        
    def set_next_settings(self, high_vram_mode: bool):
        """次回のロード時に使用する設定をセット（即時のリロードは行わない）"""
        self.next_state['high_vram'] = high_vram_mode
        print(f"次回のtext_encoder設定を設定しました: High-VRAM mode: {high_vram_mode}")
    
    def _needs_reload(self):
        """現在の状態と次回の設定を比較し、リロードが必要かどうかを判断"""
        if not self._is_loaded():
            return True
        if self.current_state['high_vram'] != self.next_state['high_vram']:
            return True
        return False
    
    def _is_loaded(self):
        """text_encoderとtext_encoder_2が読み込まれているかどうかを確認"""
        return self.text_encoder is not None and self.text_encoder_2 is not None and self.current_state['is_loaded']
    
    def get_text_encoders(self):
        """現在のtext_encoderとtext_encoder_2インスタンスを取得"""
        return self.text_encoder, self.text_encoder_2

    def dispose(self):
        """text_encoderとtext_encoder_2のインスタンスを破棄し、メモリを完全に解放"""
        try:
            if not self._is_loaded():
                return True
            
            print("text_encoderとtext_encoder_2のメモリを解放します...")
            
            if self.text_encoder is not None:
                if not self.current_state['high_vram']:
                    DynamicSwapInstaller.uninstall_model(self.text_encoder)
                self.text_encoder.to('cpu')
                del self.text_encoder
                self.text_encoder = None

            if self.text_encoder_2 is not None:
                if not self.current_state['high_vram']:
                    DynamicSwapInstaller.uninstall_model(self.text_encoder_2)
                self.text_encoder_2.to('cpu')
                del self.text_encoder_2
                self.text_encoder_2 = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.current_state['is_loaded'] = False
            print("text_encoderとtext_encoder_2のメモリ解放が完了しました")
            return True
            
        except Exception as e:
            print(f"text_encoderとtext_encoder_2のメモリ解放中にエラー: {e}")
            traceback.print_exc()
            return False

    def ensure_text_encoder_state(self):
        """text_encoderとtext_encoder_2の状態を確認し、必要に応じてリロード"""
        if self._needs_reload():
            print("text_encoderとtext_encoder_2をリロードします")
            return self._reload_text_encoders()        
        print("ロード済みのtext_encoderとtext_encoder_2を再度利用します")
        return True
    
    def _reload_text_encoders(self):
        """next_stateの設定でtext_encoderとtext_encoder_2をリロード"""
        try:
            self.dispose() # 既存のモデルを確実に解放

            from transformers import LlamaModel, CLIPTextModel
            
            print(f"Loading Text Encoders from: {self.model_path}")

            self.text_encoder = LlamaModel.from_pretrained(
                self.model_path, 
                subfolder='text_encoder', 
                torch_dtype=torch.bfloat16,
                local_files_only=True
            ).cpu()
            
            self.text_encoder_2 = CLIPTextModel.from_pretrained(
                self.model_path, 
                subfolder='text_encoder_2', 
                torch_dtype=torch.bfloat16,
                local_files_only=True
            ).cpu()
            
            self.text_encoder.eval()
            self.text_encoder_2.eval()
            
            self.text_encoder.requires_grad_(False)
            self.text_encoder_2.requires_grad_(False)
            
            if not self.next_state['high_vram']:
                print("Low VRAM mode: Applying DynamicSwapInstaller to TextEncoders...")
                DynamicSwapInstaller.install_model(self.text_encoder, device=self.device)
                DynamicSwapInstaller.install_model(self.text_encoder_2, device=self.device)
            else:
                print("High VRAM mode: Moving TextEncoders to GPU...")
                self.text_encoder.to(self.device)
                self.text_encoder_2.to(self.device)
            
            self.next_state['is_loaded'] = True
            self.current_state = self.next_state.copy()
            
            print("text_encoderとtext_encoder_2のリロードが完了しました")
            return True
            
        except Exception as e:
            print(f"text_encoderとtext_encoder_2リロードエラー: {e}")
            traceback.print_exc()
            self.current_state['is_loaded'] = False
            return False

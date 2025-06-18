import os
from huggingface_hub import hf_hub_download

class FramepackValidator:
    """Validate that FramePack F1 files were downloaded correctly"""

    # --- 修正点 ---
    # 検証対象のファイルリストを、実際のファイル構造に合わせて変更します。
    # "model.safetensors" の代わりに、分割モデルのインデックスファイル "diffusion_pytorch_model.safetensors.index.json" を指定します。
    FILES = [
        ("hunyuanvideo-community/HunyuanVideo", "vae/config.json"),
        ("lllyasviel/FramePack_F1_I2V_HY_20250503", "diffusion_pytorch_model.safetensors.index.json"),
    ]
    # --- 修正はここまで ---

    def validate_model_files(self, repo_id: str, file_path: str) -> bool:
        try:
            # この関数は、指定されたファイルが存在するかどうかをチェックするだけなので、ロジックの変更は不要です。
            local = hf_hub_download(repo_id=repo_id, filename=file_path)
            size = os.path.getsize(local)
            if size < 200:
                with open(local, "r") as f:
                    text = f.read()
                if text.startswith("version https://git-lfs.github.com"):
                    print(f"Validation failed for {repo_id}/{file_path}")
                    return False
            return True
        except Exception as e:
            print(f"Validation error for {repo_id}/{file_path}: {e}")
            return False

    def validate_all_components(self) -> bool:
        all_ok = True
        for repo, path in self.FILES:
            if not self.validate_model_files(repo, path):
                all_ok = False
        return all_ok

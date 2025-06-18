# 必要なライブラリをインポート
import subprocess
from pathlib import Path
import sys

class FramepackDownloader:
    """
    Download FramePack F1 repositories using the huggingface-cli command-line tool
    to ensure stability within the Forge environment.
    """

    REPOS = [
        "hunyuanvideo-community/HunyuanVideo",
        "lllyasviel/flux_redux_bfl",
        "lllyasviel/FramePack_F1_I2V_HY_20250503",
    ]

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def _run_download_command(self, repo: str):
        """
        Executes the huggingface-cli download command in a separate process.
        """
        print(f"Downloading {repo} via huggingface-cli...")
        
        # 実行するコマンドをリストとして構築
        cmd = [
            "huggingface-cli", "download",
            repo,
            "--local-dir-use-symlinks", "False"
        ]
        
        # キャッシュディレクトリが指定されている場合のみコマンドに追加
        if self.cache_dir:
            cmd.extend(["--cache-dir", str(self.cache_dir)])

        try:
            # subprocess.Popenを使い、コマンドを非同期待ち受けで実行
            # これにより、ダウンロードの進捗をリアルタイムで表示できる
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                bufsize=1,
                universal_newlines=True
            )

            # プロセスの標準出力を一行ずつ読み取り、表示する
            for line in iter(process.stdout.readline, ''):
                print(f"   > {line.strip()}")
            
            # プロセスの終了を待つ
            process.wait()

            # リターンコードが0でない場合（エラー終了）は例外を発生させる
            if process.returncode != 0:
                print(f"Error: Download process for {repo} failed with return code {process.returncode}.")
                raise subprocess.CalledProcessError(process.returncode, cmd)

        except FileNotFoundError:
            # "huggingface-cli"コマンドが見つからない場合のエラー
            print("\n" + "="*50)
            print("FATAL ERROR: `huggingface-cli` command not found.")
            print("Please ensure 'huggingface_hub' is correctly installed in your environment.")
            print(f"Attempting to install with: '{sys.executable} -m pip install -U huggingface_hub'")
            print("="*50 + "\n")
            raise
        except subprocess.CalledProcessError as e:
            # ダウンロード自体が失敗した場合のエラー
            print(f"An error occurred during the download of {repo}.")
            raise e
            
    def download_all_models(self):
        """
        Iterates through all required repositories and downloads them.
        """
        for repo in self.REPOS:
            self._run_download_command(repo)
            print(f"Finished processing download for {repo}")

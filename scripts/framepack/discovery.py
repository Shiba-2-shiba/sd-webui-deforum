import os
from pathlib import Path


class FramepackDiscovery:
    """
    FramePack F1モデルの存在を確認し、ローカルパスを解決します。
    環境変数 ``HF_HOME`` の値だけでなく、Hugging Face 標準のキャッシュパス
    (``~/.cache/huggingface``) も常に探索対象に含めることで、環境設定に依存しない
    より堅牢な探索ロジックを提供します。デバッグログも詳細に出力されます。
    """

    REQUIRED_COMPONENTS = {
        "transformer": {
            "repo_id": "lllyasviel/FramePack_F1_I2V_HY_20250503",
            "check_file": "diffusion_pytorch_model.safetensors.index.json",
        },
        "text_encoder": {
            "repo_id": "hunyuanvideo-community/HunyuanVideo",
            "check_file": "text_encoder/config.json",
        },
        "vae": {
            "repo_id": "hunyuanvideo-community/HunyuanVideo",
            "check_file": "vae/config.json",
        },
        "flux_bfl": {
            "repo_id": "lllyasviel/flux_redux_bfl",
            "check_file": "model_index.json",
        },
    }

    def __init__(self, cache_dir: str | None = None):
        print("\n" + "=" * 50)
        print("--- [Discovery] Initializing... ---")

        # ステップ1：ベース候補となるキャッシュディレクトリを収集
        base_candidates = []
        if cache_dir:
            base_candidates.append(Path(cache_dir))
        if os.getenv("HF_HOME"):
            base_candidates.append(Path(os.getenv("HF_HOME")))
        base_candidates.append(Path.home() / ".cache/huggingface")

        print("[DEBUG] Base cache candidates:")
        for c in base_candidates:
            print(f"  - {c}")

        # ステップ2：各候補を最終的な探索ディレクトリリストに変換
        self.hub_caches = []
        for base_cache_dir in base_candidates:
            if (base_cache_dir / "hub").is_dir():
                hub_dir = base_cache_dir / "hub"
                self.hub_caches.append(hub_dir)
                print(
                    f"[DEBUG] 'hub' directory found in {base_cache_dir}. Adding {hub_dir}"
                )
            else:
                self.hub_caches.append(base_cache_dir)
                print(
                    f"[DEBUG] 'hub' directory not found in {base_cache_dir}. Adding {base_cache_dir}"
                )

        # 重複排除しつつ順序維持
        unique = []
        seen = set()
        for p in self.hub_caches:
            if p not in seen:
                unique.append(p)
                seen.add(p)
        self.hub_caches = unique

        print("[INFO] Final resolved cache directories to be searched:")
        for idx, path in enumerate(self.hub_caches, start=1):
            print(f"  {idx}. {path} (exists: {path.is_dir()})")
        print("=" * 50 + "\n")

    def _get_snapshot_path(self, repo_id: str) -> Path | None:
        """
        指定されたリポジトリの最新スナップショットディレクトリパスを柔軟に探索して取得する。
        """
        repo_path = None
        print(f"  [search] Searching for repository '{repo_id}'...")
        for hub_cache in self.hub_caches:
            possible_paths = [
                # 1. Hugging Face標準の命名規則 (例: models--user--repo)
                hub_cache / f"models--{repo_id.replace('/', '--')}",
                # 2. リポジトリ名のみの単純な命名規則 (例: HunyuanVideo)
                hub_cache / repo_id.split("/")[-1],
            ]

            for path_to_check in possible_paths:
                print(f"  -> Probing path: {path_to_check}")
                if path_to_check.is_dir():
                    repo_path = path_to_check
                    print(f"  [search] OK: Repository found at probed path.")
                    break
            if repo_path:
                break

        if not repo_path:
            print(
                f"  [search] FAIL: Repository directory not found in any known location."
            )
            return None

        # snapshots ディレクトリを探索
        snapshots_dir = repo_path / "snapshots"
        print(f"  [sub-check] Attempting to find snapshots directory: {snapshots_dir}")
        if not snapshots_dir.is_dir():
            print(
                f"  [sub-check] INFO: 'snapshots' directory not found. Assuming the repo path itself is the snapshot."
            )
            # Fallback: snapshotsディレクトリがない場合、リポジトリルートをスナップショットパスとして返す
            return repo_path

        print(f"  [sub-check] OK: Snapshots directory found.")

        try:
            # 最新のスナップショット（更新日時が最も新しいディレクトリ）を選択
            latest_snapshot = max(
                snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime
            )
            print(f"  [sub-check] OK: Found latest snapshot: {latest_snapshot.name}")
            return latest_snapshot
        except ValueError:
            print(
                f"  [sub-check] FAIL: No snapshot directories found inside {snapshots_dir}."
            )
            return None

    def check_models_exist(self) -> tuple[bool, list[str]]:
        """全ての必須モデルが存在するかチェックする。"""
        missing_repos = []
        all_found = True

        for component, info in self.REQUIRED_COMPONENTS.items():
            repo_id = info["repo_id"]
            check_file = info["check_file"]

            # 同じリポジトリを何度もチェックしないための最適化
            if repo_id in missing_repos:
                continue

            print(f"\n--- Checking component: '{component}' (Repo: {repo_id}) ---")

            snapshot_path = self._get_snapshot_path(repo_id)

            if snapshot_path:
                full_check_file_path = snapshot_path / check_file
                print(
                    f"  [Final Check] Verifying existence of file: {full_check_file_path}"
                )

                if full_check_file_path.exists():
                    print(f"  [Result] SUCCESS: Required file found.")
                else:
                    print(f"  [Result] FAIL: Required file NOT found.")
                    all_found = False
                    if repo_id not in missing_repos:
                        missing_repos.append(repo_id)
            else:
                print(
                    f"  [Result] FAIL: Snapshot directory for this component not found."
                )
                all_found = False
                if repo_id not in missing_repos:
                    missing_repos.append(repo_id)

        print("\n--- All checks complete. ---")
        return all_found, missing_repos

    def get_local_path(self, component_name: str) -> str | None:
        """指定されたコンポーネントのローカルパス（snapshotディレクトリ）を取得する"""
        if component_name not in self.REQUIRED_COMPONENTS:
            return None

        repo_id = self.REQUIRED_COMPONENTS[component_name]["repo_id"]
        # このメソッドは探索ロジックを再利用するだけなので、ログはcheck_models_existで十分
        # より静かな実行のために、ここではprintを省略
        snapshot_path = self._get_snapshot_path(repo_id)
        return str(snapshot_path) if snapshot_path and snapshot_path.is_dir() else None

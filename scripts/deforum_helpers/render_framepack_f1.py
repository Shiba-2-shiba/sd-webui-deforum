# render_framepack_f1.py (最終修正版)

"""Entry point for FramePack F1 rendering using integration layer."""

import os
from scripts.framepack.integration import FramepackIntegration
from modules import shared


def render_animation_f1(args, anim_args, video_args, framepack_f1_args, root):
    """Generate video using FramePack F1 via the integration helper."""

    print("Forcing Hugging Face Hub to OFFLINE mode for local model loading.")
    os.environ['HF_HUB_OFFLINE'] = '1'

    integration = FramepackIntegration(device=root.device)
    # ★★★ 修正箇所1: tryブロックの戻り値を格納する変数を宣言 ★★★
    returned_data = None
    try:
        integration.setup_environment(shared.sd_model)
        # ★★★ 修正箇所2: integration.generate_videoの戻り値をreturned_dataに格納 ★★★
        returned_data = integration.generate_video(args, anim_args, video_args, framepack_f1_args, root)
    finally:
        integration.cleanup_environment()

        if 'HF_HUB_OFFLINE' in os.environ:
            print("Restoring Hugging Face Hub to ONLINE mode.")
            del os.environ['HF_HUB_OFFLINE']
            
    # ★★★ 修正箇所3: UIに結果を返すため、Noneではなく格納したデータを返す ★★★
    return returned_data

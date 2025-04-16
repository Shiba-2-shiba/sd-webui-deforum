# deforum_controlnet_gradio.py の修正案 (sd-webui-forge-classic 対応)
# このスクリプトは、DeforumのControlNet UI要素の表示/非表示を制御する
# Gradioヘルパー関数を提供します。
# sd-webui-forge環境に合わせて不要な部分は削除されています。

import gradio as gr

# --- UI 要素の表示/非表示を制御するヘルパー関数 ---
# これらの関数は deforum_controlnet.py のUIイベントハンドラから呼ばれます。

def hide_ui_by_cn_status(choice):
    """
    ControlNetユニットの有効/無効チェックボックスの状態に基づいて、
    関連するUI要素の表示/非表示を切り替えます。

    Args:
        choice (bool): チェックボックスの状態 (True: 有効, False: 無効)

    Returns:
        gradio.update: 表示状態を更新するためのGradio updateオブジェクト
    """
    # チェックボックスがTrueなら表示(visible=True)、Falseなら非表示(visible=False)
    return gr.update(visible=bool(choice))

def hide_file_textboxes(choice):
    """
    ControlNetユニットのLoopback Modeチェックボックスの状態に基づいて、
    ファイル入力（動画/画像パス）テキストボックスの表示/非表示を切り替えます。

    Args:
        choice (bool): Loopback Modeチェックボックスの状態 (True: ループバック有効, False: 無効)

    Returns:
        gradio.update: 表示状態を更新するためのGradio updateオブジェクト
    """
    # Loopback ModeがTrueの場合はファイル入力を非表示(visible=False)にするため、
    # hide_ui_by_cn_statusとは逆のロジックになります。
    return gr.update(visible=not bool(choice))

# --- 不要になったコード ---
# 元のA1111版Deforumに含まれていた可能性のある以下の要素は、
# Forge版への対応で不要になったため削除されています。
# - ToolButton クラス定義: lib_controlnet.controlnet_ui.tool_button からインポートするため不要。
# - デバッグ用のテーブル表示コードなど。

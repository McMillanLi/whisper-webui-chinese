import os
import json
import torch
import whisper
import gradio as gr

# 模型缓存目录设为项目内文件夹
os.environ["XDG_CACHE_HOME"] = os.path.join(os.getcwd(), "models")

CONFIG_PATH = "config.json"
default_config = {
    "model_size": "base",
    "save_formats": [],
    "save_dir": "",
    "device": "CPU"
}

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return default_config.copy()

def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

def on_model_change(choice):
    cfg = load_config()
    cfg["model_size"] = choice
    save_config(cfg)

def on_format_change(choices):
    cfg = load_config()
    cfg["save_formats"] = choices
    save_config(cfg)

def on_device_change(choice):
    cfg = load_config()
    cfg["device"] = choice
    save_config(cfg)

def on_save_dir_confirm(new_path):
    cfg = load_config()
    cfg["save_dir"] = new_path
    save_config(cfg)
    return f"保存目录设置为：{new_path}"

def open_save_folder():
    cfg = load_config()
    path = cfg.get("save_dir", "")
    if os.path.isdir(path):
        os.startfile(path)
        return f"打开目录：{path}"
    else:
        return "❌ 保存目录无效或不存在"

def format_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def save_transcription(result, audio_path, output_dir, formats):
    filename = os.path.splitext(os.path.basename(audio_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    if "txt" in formats:
        with open(os.path.join(output_dir, f"{filename}.txt"), "w", encoding="utf-8") as f:
            f.write(result["text"])

    if "json" in formats:
        with open(os.path.join(output_dir, f"{filename}.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    if "srt" in formats:
        with open(os.path.join(output_dir, f"{filename}.srt"), "w", encoding="utf-8") as f:
            for i, segment in enumerate(result["segments"]):
                f.write(f"{i+1}\n")
                start = format_srt_time(segment["start"])
                end = format_srt_time(segment["end"])
                text = segment["text"].strip()
                f.write(f"{start} --> {end}\n{text}\n\n")

    if "vtt" in formats:
        with open(os.path.join(output_dir, f"{filename}.vtt"), "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for segment in result["segments"]:
                start = format_srt_time(segment["start"])
                end = format_srt_time(segment["end"])
                text = segment["text"].strip()
                f.write(f"{start} --> {end}\n{text}\n\n")

    if "tsv" in formats:
        with open(os.path.join(output_dir, f"{filename}.tsv"), "w", encoding="utf-8") as f:
            f.write("start\tend\ttext\n")
            for segment in result["segments"]:
                f.write(f"{segment['start']}\t{segment['end']}\t{segment['text'].strip()}\n")

# 缓存模型对象
loaded_models = {}

def get_model():
    cfg = load_config()
    device = "cuda" if cfg["device"] == "GPU" and torch.cuda.is_available() else "cpu"
    model_key = f"{cfg['model_size']}-{device}"
    if model_key not in loaded_models:
        model = whisper.load_model(cfg["model_size"])
        model.to(device)
        loaded_models[model_key] = model
    return loaded_models[model_key], device

def transcribe(audio_file):
    if not audio_file:
        return "请上传一个音频文件。"

    cfg = load_config()
    model, _ = get_model()
    result = model.transcribe(audio_file, language="zh")

    if cfg["save_formats"] and cfg["save_dir"]:
        save_transcription(result, audio_file, cfg["save_dir"], cfg["save_formats"])

    return result["text"]

def batch_transcribe(folder_path):
    if not os.path.isdir(folder_path):
        return "❌ 目录无效，请输入正确的路径"

    cfg = load_config()
    model, _ = get_model()
    exts = [".mp3", ".wav", ".m4a", ".flac", ".ogg"]
    results = []
    files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in exts]

    if not files:
        return "⚠️ 文件夹中没有音频文件"

    for f in files:
        path = os.path.join(folder_path, f)
        try:
            result = model.transcribe(path, language="zh")
            if cfg["save_formats"] and cfg["save_dir"]:
                save_transcription(result, path, cfg["save_dir"], cfg["save_formats"])
            results.append(f"✅ 成功：{f}")
        except Exception as e:
            results.append(f"❌ 失败：{f}，原因：{str(e)}")

    return "\n".join(results)

# ---------- 构建界面 ----------
cfg = load_config()

with gr.Blocks(title="Whisper 中文转写 WebUI") as demo:
    gr.Markdown("## 🎙 Whisper 中文转写界面")

    with gr.Row():
        # 左侧设置区
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ 设置")

            model_size = gr.Radio(["tiny", "base", "small", "medium", "large"],
                                  label="模型大小", value=cfg["model_size"])
            model_size.change(fn=on_model_change, inputs=model_size)

            save_formats = gr.CheckboxGroup(["txt", "srt", "vtt", "json", "tsv"],
                                            label="保存格式", value=cfg["save_formats"])
            save_formats.change(fn=on_format_change, inputs=save_formats)

            device_select = gr.Radio(["CPU", "GPU"], label="运行设备", value=cfg["device"])
            device_select.change(fn=on_device_change, inputs=device_select)

            gr.Markdown("#### 📁 保存目录")
            save_dir_input = gr.Textbox(value=cfg["save_dir"], label="路径")
            confirm_btn = gr.Button("✅ 确认保存目录")
            open_folder_btn = gr.Button("📂 打开保存目录")
            save_dir_status = gr.Textbox(value="", label="状态", interactive=False)

            confirm_btn.click(fn=on_save_dir_confirm, inputs=save_dir_input, outputs=save_dir_status)
            open_folder_btn.click(fn=open_save_folder, outputs=save_dir_status)

        # 右侧输出区：标签页
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("🎙 单文件转写"):
                    audio_input = gr.Audio(type="filepath", label="上传音频文件")
                    transcribe_btn = gr.Button("开始转写")
                    result_text = gr.Textbox(label="转写结果", lines=10)
                    transcribe_btn.click(fn=transcribe, inputs=audio_input, outputs=result_text)

                with gr.Tab("📂 批量转写"):
                    folder_input = gr.Textbox(label="输入音频文件夹路径")
                    batch_btn = gr.Button("开始批量转写")
                    batch_output = gr.Textbox(label="批量转写状态", lines=15)
                    batch_btn.click(fn=batch_transcribe, inputs=folder_input, outputs=batch_output)

# 启动
if __name__ == "__main__":
    demo.launch()

# 🎙 Whisper 中文转写 WebUI

一个基于 OpenAI Whisper 的轻量中文语音转写工具，提供直观的 Web 界面，支持 GPU/CPU 选择、转写结果多格式导出、批量转写等实用功能。

## 🚀 功能亮点

- ✅ 支持 Whisper 全模型（tiny/base/small/medium/large）
- ✅ 中文转写（强制语言为 zh）
- ✅ GPU/CPU 运行设备自由切换
- ✅ 多格式保存（txt, srt, vtt, json, tsv）
- ✅ 批量转写：支持整个文件夹一键处理
- ✅ 自动保存配置：模型、设备、保存格式与路径
- ✅ 模型文件缓存在项目本地，不占用系统盘空间

---

## 📦 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/yourname/whisper-webui-chinese.git
cd whisper-webui-chinese
```
### 2. 创建 Conda 环境（可选）
```bash
conda create -n whisper python=3.10 -y
conda activate whisper
```
### 3. 安装依赖
```bash
pip install -r requirements.txt
```
### 4. 启动 WebUI
```bash
python app.py
打开浏览器访问：http://127.0.0.1:7860
```

## 📁 项目结构
```bash
.
├── app.py            # 主程序入口
├── config.json       # 自动生成的配置文件
├── models/           # Whisper 模型本地缓存目录（避免写入 C 盘）
├── requirements.txt  # 所需依赖
└── README.md         # 本文件
```
## 📷 界面预览
| 单文件转写                                             | 批量转写                                            |
| ------------------------------------------------- | ----------------------------------------------- |
| ![single](https://your.screenshot.url/single.png) | ![batch](https://your.screenshot.url/batch.png) |


## 🧠 TODO
 增加语音自动识别语言功能

 实时字幕预览

 文件重复检测（避免重复转写）


# video watermark removal/批量视频去水印

watermark detection by paddle OCR + watermark removal by ProPainter
batch processing videos

# Download & Install:
git clone https://github.com/FatTsubasa/VideoRemoveWatermark.git

cd VideoRemoveWatermark

python -m venv venv
**Windows cmd**
venv\Scripts\activate
**Windows powershell**
.\venv\Scripts\Activate.ps1
**Linux/macOS**
source venv/bin/activate

pip install -r requirements.txt

**download Propainter project:**
git clone https://github.com/sczhou/ProPainter.git

**install GPU related paddle:**
http://www.paddleocr.ai/latest/version3.x/installation.html

such as I need to run on NV50x GPU
python -m pip install https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-TagBuild-Training-Windows-Gpu-Cuda12.9-Cudnn9.9-Trt10.5-Mkl-Avx-VS2019-SelfBuiltPypiUse/86d658f56ebf3a5a7b2b33ace48f22d10680d311/paddlepaddle_gpu-3.0.0.dev20250717-cp312-cp312-win_amd64.whl --timeout 300 --retries 5

when run first time, Propainter will download weight files automatic but it may be very slow
better to download below manually then put them in weights folder (https://github.com/sczhou/ProPainter/releases/tag/v0.1.0)

recurrent_flow_completion.pth
raft-things.pth
ProPainter.pth

# How to use:
1.put all your videos in input_videos
2..\env\Scripts\Activate.ps1 (power shell)
3.python run_win.py
4.all watermark removed video generated in output_videos

Reference:
http://www.paddleocr.ai/latest/index.html
https://github.com/sczhou/ProPainter


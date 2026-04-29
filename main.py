#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
import argparse
import base64
import logging
import os
import time
import uuid
from typing import Optional
from urllib.parse import urlparse

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.mimo_audio.mimo_audio import MimoAudio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="MiMo-V2.5-ASR API Service")

service_ready = False
service_model: Optional[MimoAudio] = None
model_dir = None

ASSET_DIR = '/home/workspace/music-asr/asset'

LANGUAGE_TAGS = {
    "auto": "",
    "chinese": "<chinese>",
    "english": "<english>",
}


class ASRRequest(BaseModel):
    taskId: str
    audioUrl: Optional[str] = None
    audioBase64: Optional[str] = None
    language: str = "auto"  # auto / chinese / english


@app.on_event("startup")
async def startup_event():
    global service_ready, service_model, model_dir

    logger.info("正在启动 MiMo-V2.5-ASR API 服务...")

    try:
        if not model_dir:
            raise ValueError("model_dir 未设置，请通过 --model_dir 参数指定模型根目录")

        _model_path = os.path.join(model_dir, "MiMo-V2.5-ASR")
        _tokenizer_path = os.path.join(model_dir, "MiMo-Audio-Tokenizer")

        logger.info(f"加载模型: {_model_path}")
        logger.info(f"加载 tokenizer: {_tokenizer_path}")

        service_model = MimoAudio(_model_path, _tokenizer_path)

        service_ready = True
        logger.info("MiMo-V2.5-ASR API 服务启动完成，服务已就绪！")

    except Exception as e:
        logger.error(f"初始化失败: {e}", exc_info=True)
        service_ready = False
        raise


@app.get("/health")
async def health_check():
    if service_ready:
        return {"status": "healthy", "ready": True, "message": "服务已就绪"}
    return {"status": "initializing", "ready": False, "message": "服务正在初始化中"}


def _audio_tag(language: str) -> str:
    return LANGUAGE_TAGS.get(language.lower(), "")


def process_asr_url(task_id: str, audio_url: str, language: str) -> dict:
    global service_model

    total_start = time.time()
    parsed = urlparse(audio_url)
    filename = os.path.basename(parsed.path) or f"{task_id}.mp3"
    name, ext = os.path.splitext(filename)
    if not ext:
        filename = f"{task_id}_{name}.mp3"
    audio_path = os.path.join(ASSET_DIR, filename)

    logger.info(f"开始从 URL 下载音频: {audio_url}")
    dl_start = time.time()
    response = requests.get(audio_url, timeout=300, stream=True)
    response.raise_for_status()
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    with open(audio_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=262144):
            if chunk:
                f.write(chunk)
    download_time = time.time() - dl_start
    logger.info(f"下载完成: {audio_path}, 耗时: {download_time:.3f}s")

    try:
        audio_tag = _audio_tag(language)
        logger.info(f"开始识别: {audio_path}, language={language}, tag='{audio_tag}'")
        res = service_model.asr_sft(audio_path, audio_tag=audio_tag)
        logger.info(f"识别完成: {audio_path}")
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    return {
        "success": True,
        "taskId": task_id,
        "audioUrl": audio_url,
        "message": "",
        "results": [res],
        "costTime": time.time() - total_start,
        "audioDownloadTime": download_time,
    }


def process_asr_base64(task_id: str, audio_base64: str, language: str) -> dict:
    global service_model

    total_start = time.time()
    audio_path = os.path.join(ASSET_DIR, f"{uuid.uuid4().hex}.mp3")

    logger.info("开始解析 base64 音频数据")
    dl_start = time.time()
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    audio_bytes = base64.b64decode(audio_base64)
    with open(audio_path, 'wb') as f:
        f.write(audio_bytes)
    decode_time = time.time() - dl_start
    logger.info(f"base64 解析完成: {audio_path}, 耗时: {decode_time:.3f}s")

    try:
        audio_tag = _audio_tag(language)
        logger.info(f"开始识别: {audio_path}, language={language}, tag='{audio_tag}'")
        res = service_model.asr_sft(audio_path, audio_tag=audio_tag)
        logger.info(f"识别完成: {audio_path}")
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    return {
        "success": True,
        "taskId": task_id,
        "audioUrl": None,
        "message": "",
        "results": [res],
        "costTime": time.time() - total_start,
        "audioDownloadTime": decode_time,
    }


@app.post("/asr")
async def asr_recognize(request: ASRRequest = Body(...)):
    global service_ready

    if not service_ready:
        raise HTTPException(status_code=503, detail="服务正在初始化中，请稍后再试")

    logger.info(f"收到 ASR 请求: taskId={request.taskId}")

    task_id = request.taskId
    if not task_id or not task_id.strip():
        raise HTTPException(status_code=400, detail="taskId 参数不能为空")

    audio_url = request.audioUrl if request.audioUrl and request.audioUrl.strip() else None
    audio_base64 = request.audioBase64 if request.audioBase64 and request.audioBase64.strip() else None

    if audio_url is None and audio_base64 is None:
        raise HTTPException(status_code=400, detail="未传入有效的音频 URL 或 base64 编码数据")

    try:
        if audio_url is not None:
            result = process_asr_url(task_id, audio_url, request.language)
        else:
            result = process_asr_base64(task_id, audio_base64, request.language)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"ASR 处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MiMo-V2.5-ASR API Service')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务主机地址')
    parser.add_argument('--port', type=int, default=8000, help='服务端口号')
    parser.add_argument('--model_dir', type=str, required=True, help='模型根目录，需包含 MiMo-V2.5-ASR 和 MiMo-Audio-Tokenizer 子目录')

    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        logger.error(f"model_dir 不存在: {args.model_dir}")
        raise ValueError(f"model_dir 不存在: {args.model_dir}")

    for sub in ("MiMo-V2.5-ASR", "MiMo-Audio-Tokenizer"):
        sub_path = os.path.join(args.model_dir, sub)
        if not os.path.exists(sub_path):
            logger.error(f"子目录不存在: {sub_path}")
            raise ValueError(f"子目录不存在: {sub_path}")

    model_dir = args.model_dir
    logger.info(f"model_dir: {model_dir}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

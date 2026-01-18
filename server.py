import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.agent.manus import Manus
from app.logger import logger
from app.schema import AgentState


class PromptRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for the agent")


class PromptResponse(BaseModel):
    status: str
    message: str


agent: Optional[Manus] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    logger.info("Starting Manus agent...")
    agent = await Manus.create()
    logger.info("Manus agent initialized successfully")
    yield
    logger.info("Shutting down Manus agent...")
    if agent:
        await agent.cleanup()
    logger.info("Manus agent shutdown complete")


app = FastAPI(
    title="OpenManus API",
    description="A versatile agent API that can solve various tasks using multiple tools",
    version="0.1.0",
    lifespan=lifespan,
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境应限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有 HTTP 头
)


@app.get("/")
async def read_root():
    return FileResponse("web_frontend.html")


@app.get("/")
async def root():
    return {
        "message": "OpenManus API is running",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "run": "/run",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "agent_initialized": agent is not None,
    }


@app.post("/run", response_model=PromptResponse)
async def run_agent(request: PromptRequest):
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Empty prompt provided")

    try:
        await agent.cleanup()
        agent.current_step = 0
        agent.state = AgentState.IDLE
        agent.memory.clear()
        logger.info(f"Processing request: {request.prompt[:100]}...")
        await agent.run(request.prompt)
        logger.info("Request processing completed")
        return PromptResponse(status="success", message="Request processing completed")
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )


@app.get("/get-latest-report")
async def get_latest_report():
    import os
    from pathlib import Path

    workspace = Path("workspace")
    if not workspace.exists() or not workspace.is_dir():
        raise HTTPException(status_code=404, detail="Workspace directory not found")

    # 获取 workspace 下最新的文件
    try:
        latest_file = max(
            workspace.iterdir(), key=lambda f: f.stat().st_mtime if f.is_file() else 0
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="No files found in workspace")

    # 读取并返回文件内容
    try:
        content = latest_file.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        raise HTTPException(status_code=500, detail="Failed to read latest report")

    return {"filename": latest_file.name, "content": content}


import asyncio
import os
from datetime import datetime
from pathlib import Path

import aiofiles
from fastapi.responses import StreamingResponse

LOG_DIR = Path("logs")


async def tail_file(file_path: Path, last_size: int = 0):
    """流式读取文件新增内容"""
    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            await f.seek(last_size)
            while True:
                line = await f.readline()
                if line:
                    yield f"data: {line}\n\n"
                else:
                    await asyncio.sleep(0.5)
    except Exception as e:
        yield f"data: [ERROR] {e}\n\n"


@app.get("/streaming_log")
async def streaming_log():
    if not LOG_DIR.exists() or not LOG_DIR.is_dir():
        raise HTTPException(status_code=404, detail="日志目录不存在")

    # 获取 logs 目录下最新的文件
    try:
        latest_log = max(
            LOG_DIR.iterdir(), key=lambda f: f.stat().st_mtime if f.is_file() else 0
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="日志目录中没有文件")

    async def event_stream():
        async for chunk in tail_file(latest_log):
            yield chunk

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

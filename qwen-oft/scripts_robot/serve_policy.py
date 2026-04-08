import argparse
import logging
import io
from typing import Any, Dict, List, Optional

from PIL import Image

import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# 复用 LIBERO 测试脚本中的模型加载与推理逻辑
from scripts_libero.utils import (
    load_and_merge_training_config,
    model_load,
    model_predict,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 可选的 debugpy 支持
try:
    import debugpy

    DEBUGPY_AVAILABLE = True
except ImportError:
    debugpy = None
    DEBUGPY_AVAILABLE = False


class PredictRequest(BaseModel):
    task_description: str
    # 图片在 JSON 中以 base64 字符串传输，由服务端手动解码为 bytes
    images: List[str]
    state: Optional[List[float]] = None


class PredictResponse(BaseModel):
    actions: List[List[float]]


def build_model(args: argparse.Namespace) -> Dict[str, Any]:
    """
    按 test_libero_no_gen_encoder.py 的方式加载模型与统计量，并返回 model_components。
    """
    # 从 training_config.json 合并训练时配置（use_dinov3 / action_dim / action_chunk / bins 等）
    args = load_and_merge_training_config(args)

    # 加载 Qwen3VL + 连续动作头 + 统计量
    model_components = model_load(args)
    logger.info("Policy model loaded for serving.")
    return model_components


def create_app(args: argparse.Namespace, model_components: Dict[str, Any]) -> FastAPI:
    app = FastAPI(title="Qwen-Robot Policy Server")

    # 固定在内存中复用
    vl_gpt = model_components["vl_gpt"]
    processor = model_components["processor"]
    action_tokenizer = model_components["action_tokenizer"]
    action_head = model_components["action_head"]
    statistic = model_components["statistic"]
    dinov3_model = model_components["dinov3_model"]
    dinov3_processor = model_components["dinov3_processor"]
    dinov3_qwen_mapping = model_components["dinov3_qwen_mapping"]

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        if not req.images:
            raise HTTPException(status_code=400, detail="images 不能为空")

        # 取第一张图片（base64 字符串），解码为字节后解析为 PIL.Image
        import base64

        try:
            # logger.info(f"Received image string length: {len(req.images[0])}")
            img_bytes = base64.b64decode(req.images[0])
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无法解析图片数据: {e}")

        state_array: Optional[np.ndarray] = None
        if req.state is not None:
            state_array = np.array(req.state, dtype=np.float32)

        try:
            actions = model_predict(
                args=args,
                vl_gpt=vl_gpt,
                processor=processor,
                action_tokenizer=action_tokenizer,
                action_head=action_head,
                statistic=statistic,
                task_description=req.task_description,
                image=[pil_img],
                state=state_array,
                pointcloud=None,
                pre_image_dir="",
                step=0,
                dinov3_model=dinov3_model,
                dinov3_processor=dinov3_processor,
                dinov3_qwen_mapping=dinov3_qwen_mapping,
            )
        except Exception as e:
            logger.exception("模型推理失败")
            raise HTTPException(status_code=500, detail=f"模型推理失败: {e}")

        actions_np = np.array(actions)
        if actions_np.ndim == 1:
            actions_np = actions_np.reshape(1, -1)

        return PredictResponse(actions=actions_np.tolist())

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Qwen-Robot policy over HTTP")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="模型 checkpoint 目录，需包含 training_config.json / action_head.pt / stats_data.json",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="使用的 CUDA 设备编号，例如 0",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务监听 IP（默认 0.0.0.0）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务监听端口（默认 8000）",
    )
    parser.add_argument(
        "--use-robot-state",
        type=int,
        default=None,
        help="是否在推理中使用机器人状态（若不指定，则从 training_config.json 读取）",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        help="是否启用 debugpy 调试模式（0 关闭，1 开启）",
    )
    parser.add_argument(
        "--debug-port",
        type=int,
        default=5678,
        help="debugpy 监听端口（仅在 --debug=1 时生效）",
    )
    args = parser.parse_args()

    # test_libero_no_gen_encoder.py 里使用 args.cuda
    setattr(args, "cuda", args.cuda)
    if args.use_robot_state is not None:
        setattr(args, "use_robot_state", int(args.use_robot_state))

    # dataset_name 在 stats_data.json 中的 key，机器人数据一般沿用 'libero' 或训练时设置的名称
    if not hasattr(args, "dataset_name"):
        setattr(args, "dataset_name", "libero")

    return args


if __name__ == "__main__":
    args = parse_args()

    # 若开启 debug 模式，启动 debugpy
    if getattr(args, "debug", 0) == 1:
        if DEBUGPY_AVAILABLE:
            logger.info(f"Starting debugpy on 0.0.0.0:{args.debug_port}，等待调试器连接...")
            try:
                debugpy.listen(("0.0.0.0", args.debug_port))
                debugpy.wait_for_client()
                logger.info("debugpy 已连接调试器，继续启动服务。")
            except Exception as e:
                logger.warning(f"启动 debugpy 失败，继续无调试模式运行: {e}")
        else:
            logger.warning("请求启用 debug 模式，但未安装 debugpy。请先 pip install debugpy。")

    model_components = build_model(args)
    app = create_app(args, model_components)

    uvicorn.run(app, host=args.host, port=args.port)


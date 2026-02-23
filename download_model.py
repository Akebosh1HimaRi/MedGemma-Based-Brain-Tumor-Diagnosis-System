"""
从 Hugging Face 下载 MedGemma 1.5 4B 模型到本地
"""

import os
import sys
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_model_with_cli():
    """使用 Hugging Face CLI 下载模型"""
    logger.info("使用 Hugging Face CLI 下载模型...")

    # 检查是否安装了 huggingface-hub
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("未安装 huggingface-hub，请先运行: pip install huggingface-hub")
        return False

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        local_dir = os.path.join(base_dir, "models/medgemma-1.5-4b-it")

        logger.info(f"下载模型到: {local_dir}")
        logger.info("这可能需要几分钟，请耐心等待...")

        # 下载模型
        snapshot_download(
            repo_id="google/medgemma-1.5-4b-it",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )

        logger.info(f"✓ 模型下载成功！")
        logger.info(f"模型位置: {local_dir}")

        # 列出下载的文件
        logger.info("\n下载的文件:")
        for root, dirs, files in os.walk(local_dir):
            level = root.replace(local_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            logger.info(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                file_size = os.path.getsize(os.path.join(root, file)) / (1024 * 1024)
                logger.info(f'{subindent}{file} ({file_size:.2f} MB)')

        return True

    except Exception as e:
        logger.error(f"下载失败: {e}")
        return False

def download_model_with_transformers():
    """使用 transformers 库下载模型"""
    logger.info("使用 transformers 库下载模型...")

    try:
        from transformers import AutoModel, AutoProcessor, AutoImageProcessor
    except ImportError:
        logger.error("未安装 transformers，请先运行: pip install transformers")
        return False

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        local_dir = os.path.join(base_dir, "models/medgemma-1.5-4b-it")

        logger.info(f"下载模型到: {local_dir}")
        logger.info("这可能需要几分钟，请耐心等待...")

        # 创建目录
        os.makedirs(local_dir, exist_ok=True)

        # 下载模型
        logger.info("下载模型权重...")
        model = AutoModel.from_pretrained(
            "google/medgemma-1.5-4b-it",
            trust_remote_code=True,
            torch_dtype="float16"
        )
        model.save_pretrained(local_dir)

        # 下载处理器
        logger.info("下载处理器...")
        processor = AutoProcessor.from_pretrained("google/medgemma-1.5-4b-it")
        processor.save_pretrained(local_dir)

        # 下载图像处理器
        logger.info("下载图像处理器...")
        try:
            image_processor = AutoImageProcessor.from_pretrained("google/medgemma-1.5-4b-it")
            image_processor.save_pretrained(local_dir)
        except Exception as e:
            logger.warning(f"图像处理器下载失败（可选）: {e}")

        logger.info(f"✓ 模型下载成功！")
        logger.info(f"模型位置: {local_dir}")

        return True

    except Exception as e:
        logger.error(f"下载失败: {e}")
        return False

def verify_model(model_dir):
    """验证模型文件是否完整"""
    logger.info(f"\n验证模型文件...")

    required_files = [
        "config.json",
        "preprocessor_config.json",
        "tokenizer.json",
    ]

    # model.safetensors 或 pytorch_model.bin 二选一
    weight_files = ["model.safetensors", "pytorch_model.bin"]

    all_exist = True

    # 检查必需文件
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            logger.info(f"✓ {file}")
        else:
            logger.warning(f"✗ {file} (缺失)")
            all_exist = False

    # 检查权重文件
    weight_exists = False
    for file in weight_files:
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            logger.info(f"✓ {file}")
            weight_exists = True
            break

    if not weight_exists:
        logger.warning(f"✗ 权重文件缺失 (需要 {' 或 '.join(weight_files)})")
        all_exist = False

    if all_exist:
        logger.info("\n✓ 模型文件完整！")
        return True
    else:
        logger.warning("\n✗ 模型文件不完整，请重新下载")
        return False

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("MedGemma 1.5 4B 模型下载工具")
    logger.info("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "models/medgemma-1.5-4b-it")

    # 检查模型是否已存在
    if os.path.exists(model_dir):
        logger.info(f"\n模型已存在: {model_dir}")
        if verify_model(model_dir):
            logger.info("\n✓ 无需重新下载")
            return 0
        else:
            logger.info("\n模型文件不完整，将重新下载...")

    # 选择下载方法
    logger.info("\n选择下载方法:")
    logger.info("1. 使用 Hugging Face CLI（推荐，更快）")
    logger.info("2. 使用 transformers 库")

    try:
        choice = input("\n请选择 (1 或 2): ").strip()
    except KeyboardInterrupt:
        logger.info("\n下载已取消")
        return 1

    if choice == "1":
        success = download_model_with_cli()
    elif choice == "2":
        success = download_model_with_transformers()
    else:
        logger.error("无效的选择")
        return 1

    if success:
        # 验证模型
        verify_model(model_dir)
        logger.info("\n" + "=" * 60)
        logger.info("下载完成！")
        logger.info("=" * 60)
        logger.info("\n现在您可以运行训练脚本:")
        logger.info("  python train_model_lora.py")
        logger.info("\n或推理脚本:")
        logger.info("  python inference.py")
        return 0
    else:
        logger.error("\n下载失败，请检查网络连接或重试")
        return 1

if __name__ == "__main__":
    sys.exit(main())

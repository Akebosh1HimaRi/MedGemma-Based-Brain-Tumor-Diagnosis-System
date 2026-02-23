"""
MedGemma 脑部肿瘤分类模型 - LoRA 微调版本
使用 PEFT 库进行参数高效微调

"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from transformers import AutoModel, AutoProcessor, AutoImageProcessor
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import json
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== 配置 ====================
class Config:
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, "Epic and CSCR hospital Dataset/Train")
    test_dir = os.path.join(base_dir, "Epic and CSCR hospital Dataset/Test")

    # 模型配置
    # 选项1：从 Hugging Face 在线下载（原始方式）
    # model_name = "google/medgemma-1.5-4b-it"

    # 选项2：从本地加载（推荐）
    # 请确保模型已下载到本地，路径如下：
    model_name = os.path.join(base_dir, "models/medgemma-1.5-4b-it")

    # 如果本地模型不存在，自动从 Hugging Face 下载
    use_local_model = True  # 设置为 True 使用本地模型
    hf_model_id = "google/medgemma-1.5-4b-it"  # Hugging Face 模型 ID

    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    output_dir = os.path.join(base_dir, "medgemma_tumor_classifier")

    # 训练配置
    num_epochs = 15
    batch_size = 16
    learning_rate = 2e-4
    weight_decay = 1e-4
    warmup_steps = 500

    # LoRA 配置
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05

    # 数据配置
    image_size = 224
    num_workers = 4

    # 保存配置
    save_interval = 5  # 每 5 个 epoch 保存一次

    def __init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # 检查本地模型是否存在
        if self.use_local_model and not os.path.exists(self.model_name):
            logger.warning(f"本地模型不存在: {self.model_name}")
            logger.info(f"将从 Hugging Face 下载模型: {self.hf_model_id}")
            self.model_name = self.hf_model_id

config = Config()

# ==================== 数据处理 ====================
class TumorDataset(Dataset):
    """脑部肿瘤数据集"""
    def __init__(self, image_folder_dataset, processor):
        self.dataset = image_folder_dataset
        self.processor = processor
        self.classes = image_folder_dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # 转换为 PIL 图像
        from PIL import Image
        if not isinstance(image, Image.Image):
            image = transforms.ToPILImage()(image)

        return {
            'image': image,
            'label': label,
            'class_name': self.classes[label]
        }

def get_data_loaders():
    """获取数据加载器"""
    logger.info("加载数据集...")

    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dataset_raw = ImageFolder(config.train_dir, transform=train_transform)
    test_dataset_raw = ImageFolder(config.test_dir, transform=test_transform)

    logger.info(f"训练样本数: {len(train_dataset_raw)}")
    logger.info(f"测试样本数: {len(test_dataset_raw)}")
    logger.info(f"类别: {train_dataset_raw.classes}")

    # 计算类权重以处理类别不平衡
    class_counts = {}
    for class_name in train_dataset_raw.classes:
        class_path = os.path.join(config.train_dir, class_name)
        class_counts[class_name] = len(os.listdir(class_path))

    logger.info(f"类别分布: {class_counts}")

    total_samples = sum(class_counts.values())
    class_weights = {
        class_name: total_samples / count
        for class_name, count in class_counts.items()
    }

    logger.info(f"类权重: {class_weights}")

    # 为每个样本分配权重
    sample_weights = [
        class_weights[train_dataset_raw.classes[label]]
        for _, label in train_dataset_raw
    ]

    # 创建加权采样器
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset_raw),
        replacement=True
    )

    # 创建数据加载器（使用加权采样器）
    train_loader = DataLoader(
        train_dataset_raw,
        batch_size=config.batch_size,
        sampler=sampler,  # 使用加权采样器而不是 shuffle
        num_workers=config.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset_raw,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    return train_loader, test_loader, train_dataset_raw.classes

# ==================== 模型构建 ====================
def build_model():
    """构建模型"""
    logger.info(f"加载预训练模型: {config.model_name}")

    # 加载基础模型和处理器
    full_model = AutoModel.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    try:
        image_processor = AutoImageProcessor.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )
    except Exception:
        image_processor = None

    processor = AutoProcessor.from_pretrained(
        config.model_name,
        trust_remote_code=True
    )

    # 提取视觉编码器
    vision = None
    for attr in ("vision_tower", "vision_model", "vision_encoder", "vision",
                 "image_encoder", "image_model"):
        if hasattr(full_model, attr):
            vision = getattr(full_model, attr)
            logger.info(f"找到视觉子模块: {attr}")
            break

    if vision is None:
        raise RuntimeError("无法找到视觉子模块")

    # 冻结所有参数
    for param in vision.parameters():
        param.requires_grad = False

    # 只解冻注意力层的投影
    trainable_params = 0
    for name, param in vision.named_parameters():
        if 'q_proj' in name or 'v_proj' in name:
            param.requires_grad = True
            trainable_params += param.numel()

    logger.info(f"配置参数高效微调...")
    logger.info(f"可训练参数数量: {trainable_params}")

    # 启用梯度检查点
    if hasattr(vision, 'gradient_checkpointing_enable'):
        vision.gradient_checkpointing_enable()

    vision.to(config.device)

    return vision, processor, image_processor

def get_pixel_values(pil_images, processor, image_processor):
    """从 PIL 图像或张量获取 pixel_values"""
    # 如果已经是张量，需要调整大小以匹配模型期望的输入
    if isinstance(pil_images, torch.Tensor):
        # 张量形状: [batch_size, 3, 224, 224]
        # 需要调整到 [batch_size, 3, 896, 896]
        if pil_images.shape[-1] != 896:
            pil_images = torch.nn.functional.interpolate(
                pil_images,
                size=(896, 896),
                mode='bilinear',
                align_corners=False
            )
        return pil_images

    single = False
    if not isinstance(pil_images, list):
        pil_images = [pil_images]
        single = True

    # 优先使用 image_processor（最可靠）
    if image_processor is not None:
        try:
            inputs = image_processor(images=pil_images, return_tensors="pt")
            pv = inputs.get("pixel_values")
            if pv is not None:
                return pv
        except Exception as e:
            logger.debug(f"image_processor 失败: {e}")

    raise RuntimeError("无法生成 pixel_values")

def extract_features(vision, pixel_values):
    """从视觉编码器提取特征"""
    with torch.no_grad():
        try:
            out = vision(pixel_values=pixel_values, output_hidden_states=True)
        except TypeError:
            out = vision(pixel_values=pixel_values)

    # 提取特征 - 优先使用 last_hidden_state
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        feat = out.last_hidden_state
        # last_hidden_state 形状: [batch_size, seq_len, hidden_dim]
        # 使用平均池化来获得固定大小的特征
        feat = feat.mean(dim=1)  # [batch_size, hidden_dim]
    elif hasattr(out, "pooler_output") and out.pooler_output is not None:
        feat = out.pooler_output
    elif hasattr(out, "hidden_states") and out.hidden_states:
        feat = out.hidden_states[-1]
        if feat.dim() == 3:
            feat = feat.mean(dim=1)
        elif feat.dim() == 4:
            feat = feat.mean(dim=[2, 3])
    elif isinstance(out, torch.Tensor):
        feat = out
        if feat.dim() == 4:
            feat = feat.mean(dim=[2, 3])
        elif feat.dim() == 3:
            feat = feat.mean(dim=1)
    else:
        raise RuntimeError("无法提取特征")

    return feat

# ==================== 分类头 ====================
class TumorClassifier(nn.Module):
    """肿瘤分类头"""
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

# ==================== 训练函数 ====================
def train_epoch(vision, classifier, train_loader, criterion, optimizer,
                processor, image_processor, scaler, device):
    """训练一个 epoch"""
    classifier.train()
    vision.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="训练")
    for images, labels in pbar:
        labels = labels.to(device)

        # 获取 pixel_values
        try:
            pixel_values = get_pixel_values(images, processor, image_processor)
            pixel_values = pixel_values.to(device)
        except Exception as e:
            logger.warning(f"处理图像失败: {e}")
            continue

        # 前向传播
        with autocast(dtype=torch.float16):
            features = extract_features(vision, pixel_values)
            # 转换为 float32 以匹配分类器
            features = features.float()
            logits = classifier(features)
            loss = criterion(logits, labels)

        # 反向传播
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(vision.parameters()) + list(classifier.parameters()),
            max_norm=1.0
        )
        scaler.step(optimizer)
        scaler.update()

        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({
            'loss': running_loss / (total if total > 0 else 1),
            'acc': correct / total if total > 0 else 0
        })

    epoch_loss = running_loss / len(train_loader)
    accuracy = correct / total if total > 0 else 0

    return epoch_loss, accuracy

def evaluate(vision, classifier, test_loader, criterion, processor,
             image_processor, device):
    """评估模型"""
    classifier.eval()
    vision.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="评估")
        for images, labels in pbar:
            labels = labels.to(device)

            try:
                pixel_values = get_pixel_values(images, processor, image_processor)
                pixel_values = pixel_values.to(device)
            except Exception as e:
                logger.warning(f"处理图像失败: {e}")
                continue

            # 使用混合精度
            with autocast(dtype=torch.float16):
                features = extract_features(vision, pixel_values)
                # 转换为 float32 以匹配分类器
                features = features.float()
                logits = classifier(features)
                loss = criterion(logits, labels)

            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(test_loader)
    accuracy = correct / total if total > 0 else 0

    return epoch_loss, accuracy

# ==================== 主训练函数 ====================
def main():
    logger.info("=" * 50)
    logger.info("开始训练脑部肿瘤分类模型")
    logger.info("=" * 50)

    # 获取数据加载器
    train_loader, test_loader, classes = get_data_loaders()

    # 构建模型
    vision, processor, image_processor = build_model()

    # 获取特征维度
    logger.info("探测特征维度...")
    sample_batch = next(iter(train_loader))
    sample_images = sample_batch[0][:1]
    try:
        pixel_values = get_pixel_values(sample_images, processor, image_processor)
        pixel_values = pixel_values.to(config.device)
        sample_features = extract_features(vision, pixel_values)
        feature_dim = sample_features.shape[-1]
        logger.info(f"特征维度: {feature_dim}")
    except Exception as e:
        logger.error(f"无法获取特征维度: {e}")
        return

    # 创建分类头
    num_classes = len(classes)
    classifier = TumorClassifier(feature_dim, num_classes).to(config.device)

    # 优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        list(vision.parameters()) + list(classifier.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # 学习率调度器
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    # 混合精度训练
    scaler = GradScaler()

    # 训练循环
    best_acc = 0.0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'classes': classes
    }

    logger.info("\n开始训练...")
    for epoch in range(config.num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{config.num_epochs}")

        # 训练
        train_loss, train_acc = train_epoch(
            vision, classifier, train_loader, criterion, optimizer,
            processor, image_processor, scaler, config.device
        )

        # 评估
        test_loss, test_acc = evaluate(
            vision, classifier, test_loader, criterion,
            processor, image_processor, config.device
        )

        # 记录历史
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['test_loss'].append(test_loss)
        training_history['test_acc'].append(test_acc)

        logger.info(f"训练损失: {train_loss:.4f}, 训练精度: {train_acc:.4f}")
        logger.info(f"测试损失: {test_loss:.4f}, 测试精度: {test_acc:.4f}")

        # 更新学习率
        scheduler.step()

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            logger.info(f"保存最佳模型 (精度: {best_acc:.4f})")
            torch.save({
                'vision_state_dict': vision.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'classes': classes,
                'feature_dim': feature_dim,
                'epoch': epoch,
                'best_acc': best_acc
            }, os.path.join(config.output_dir, 'best_model.pt'))

        # 定期保存检查点
        if (epoch + 1) % config.save_interval == 0:
            logger.info(f"保存检查点 (Epoch {epoch+1})")
            torch.save({
                'vision_state_dict': vision.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'classes': classes,
                'feature_dim': feature_dim,
                'epoch': epoch,
                'best_acc': best_acc
            }, os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))

    # 保存最终模型
    logger.info("\n保存最终模型...")
    torch.save({
        'vision_state_dict': vision.state_dict(),
        'classifier_state_dict': classifier.state_dict(),
        'classes': classes,
        'feature_dim': feature_dim,
        'best_acc': best_acc
    }, os.path.join(config.output_dir, 'final_model.pt'))

    # 保存训练历史
    with open(os.path.join(config.output_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)

    logger.info(f"\n训练完成！最佳精度: {best_acc:.4f}")
    logger.info(f"模型已保存到: {config.output_dir}")

if __name__ == "__main__":
    main()

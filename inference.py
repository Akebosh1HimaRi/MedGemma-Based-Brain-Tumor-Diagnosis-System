"""
脑部肿瘤检测推理脚本
用于加载训练好的模型进行推理和概率输出
"""

import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoModel, AutoProcessor, AutoImageProcessor, TextIteratorStreamer
from transformers import StoppingCriteria, StoppingCriteriaList
import numpy as np
import logging
from typing import Dict, Tuple, List
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

class TumorDetectionModel:
    """脑部肿瘤检测模型"""

    def __init__(self, model_path: str, device: str = "cuda", base_model_path: str = None):
        """
        初始化模型

        Args:
            model_path: 模型检查点路径（训练好的分类头）
            device: 计算设备 ("cuda" 或 "cpu")
            base_model_path: 基础模型路径（本地或 Hugging Face ID）
                            如果为 None，则优先使用本地模型，否则使用 Hugging Face 模型
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = model_path

        logger.info(f"使用设备: {self.device}")
        logger.info(f"加载模型: {model_path}")

        # 加载检查点
        checkpoint = torch.load(model_path, map_location=self.device)

        # 加载预训练模型
        # 优先使用本地模型，如果不存在则从 Hugging Face 下载
        if base_model_path is None:
            # 检查本地模型是否存在
            local_model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "models/medgemma-1.5-4b-it"
            )

            if os.path.exists(local_model_path) and os.path.exists(
                os.path.join(local_model_path, "config.json")
            ):
                base_model_path = local_model_path
                logger.info(f"检测到本地模型，使用本地路径")
            else:
                base_model_path = "google/medgemma-1.5-4b-it"
                logger.info(f"未检测到本地模型，将从 Hugging Face 下载")

        logger.info(f"加载基础模型: {base_model_path}")

        self.full_model = AutoModel.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # 加载处理器
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(
                base_model_path,
                trust_remote_code=True
            )
        except Exception:
            self.image_processor = None

        self.processor = AutoProcessor.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )

        # 提取视觉编码器
        self.vision = None
        for attr in ("vision_tower", "vision_model", "vision_encoder", "vision",
                     "image_encoder", "image_model"):
            if hasattr(self.full_model, attr):
                self.vision = getattr(self.full_model, attr)
                break

        if self.vision is None:
            raise RuntimeError("无法找到视觉子模块")

        # 加载视觉编码器权重
        # 只有在checkpoint中有vision_state_dict时才加载，否则使用预训练的权重
        if 'vision_state_dict' in checkpoint:
            try:
                self.vision.load_state_dict(checkpoint['vision_state_dict'])
                logger.info("已加载checkpoint中的vision权重")
            except Exception as e:
                logger.warning(f"无法加载vision权重，使用预训练权重: {e}")
        else:
            logger.info("使用预训练的vision权重")

        self.vision.to(self.device)
        self.vision.eval()

        # 创建分类头
        feature_dim = checkpoint['feature_dim']
        num_classes = len(checkpoint['classes'])
        self.classifier = TumorClassifier(feature_dim, num_classes)
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.classifier.to(self.device)
        self.classifier.eval()

        # 保存类别信息
        self.classes = checkpoint['classes']
        self.num_classes = num_classes

        # 准备文本生成用的模型
        # 注意：full_model已经加载，可以用于文本生成
        self.full_model.eval()
        logger.info(f"模型加载成功！类别: {self.classes}")
        logger.info("已准备好文本生成功能")

    def get_pixel_values(self, pil_images):
        """从 PIL 图像获取 pixel_values"""
        single = False
        if not isinstance(pil_images, list):
            pil_images = [pil_images]
            single = True

        # 尝试 image_processor
        if self.image_processor is not None:
            try:
                inputs = self.image_processor(images=pil_images, return_tensors="pt")
                pv = inputs.get("pixel_values")
                if pv is not None:
                    return pv
            except Exception:
                pass

        # 尝试 processor
        try:
            inputs = self.processor(images=pil_images, return_tensors="pt")
            pv = inputs.get("pixel_values")
            if pv is not None:
                return pv
        except Exception:
            pass

        # 尝试带占位符的 processor
        candidates = ["<image>", "<Image>", "<img>", "[IMG]"]
        for token in candidates:
            try:
                texts = [token] * len(pil_images)
                inputs = self.processor(images=pil_images, text=texts, return_tensors="pt")
                pv = inputs.get("pixel_values")
                if pv is not None:
                    return pv
            except Exception:
                continue

        raise RuntimeError("无法生成 pixel_values")

    def extract_features(self, pixel_values):
        """从视觉编码器提取特征"""
        try:
            logger.info(f"开始提取特征，输入形状: {pixel_values.shape}")
            with torch.no_grad():
                out = self.vision(pixel_values=pixel_values)

            logger.info(f"视觉编码器输出类型: {type(out)}")

            # 如果是BaseModelOutput或其他模型输出对象，打印所有属性
            if hasattr(out, '__dict__'):
                logger.info(f"输出对象属性: {out.__dict__.keys()}")

            # 提取特征
            feat = None

            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                feat = out.pooler_output
                logger.info(f"✓ 使用 pooler_output，形状: {feat.shape}")
            elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                feat = out.last_hidden_state
                logger.info(f"✓ 使用 last_hidden_state，原始形状: {feat.shape}")
                if feat.dim() == 3:
                    feat = feat.mean(dim=1)
                    logger.info(f"  处理后形状(dim=3): {feat.shape}")
                elif feat.dim() == 4:
                    feat = feat.mean(dim=[2, 3])
                    logger.info(f"  处理后形状(dim=4): {feat.shape}")
            elif isinstance(out, torch.Tensor):
                feat = out
                logger.info(f"✓ 直接使用Tensor输出，原始形状: {feat.shape}")
                if feat.dim() == 4:
                    feat = feat.mean(dim=[2, 3])
                    logger.info(f"  处理后形状: {feat.shape}")
                elif feat.dim() == 3:
                    feat = feat.mean(dim=1)
                    logger.info(f"  处理后形状: {feat.shape}")
            else:
                logger.error(f"❌ 无法识别输出类型: {type(out)}")
                logger.error(f"输出对象所有属性: {dir(out)}")
                # 尝试找到任何tensor属性
                for attr in dir(out):
                    if not attr.startswith('_'):
                        try:
                            val = getattr(out, attr)
                            if isinstance(val, torch.Tensor):
                                logger.info(f"  找到Tensor属性: {attr}, 形状: {val.shape}")
                        except:
                            pass
                raise RuntimeError("无法提取特征")

            if feat is None:
                logger.error("❌ feat 最终为 None")
                raise RuntimeError("特征提取结果为None")

            logger.info(f"✓ 特征提取成功，最终形状: {feat.shape}")
            return feat

        except Exception as e:
            logger.error(f"❌ extract_features 异常: {e}", exc_info=True)
            raise

    def predict(self, image_path: str) -> Dict:
        """
        对单张图像进行预测

        Args:
            image_path: 图像路径

        Returns:
            包含预测结果的字典
        """
        # 加载图像
        image = Image.open(image_path).convert('RGB')

        # 预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image)

        # 获取 pixel_values
        try:
            pixel_values = self.get_pixel_values(image)
            if pixel_values is None:
                logger.error(f"pixel_values 为 None")
                raise RuntimeError("无法获取像素值")
            pixel_values = pixel_values.to(self.device)
            logger.debug(f"pixel_values 形状: {pixel_values.shape}")
        except Exception as e:
            logger.error(f"获取 pixel_values 失败: {e}")
            raise

        # 提取特征
        try:
            features = self.extract_features(pixel_values)
            if features is None:
                logger.error(f"features 为 None")
                raise RuntimeError("无法提取特征")
            logger.debug(f"features 形状: {features.shape}")

            # 确保特征和分类头的dtype一致
            # 由于分类头是float32，而vision可能是float16，需要转换
            features = features.float()  # 转换为float32
            logger.debug(f"特征dtype转换后: {features.dtype}")
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            raise

        # 分类
        with torch.no_grad():
            logits = self.classifier(features)
            probabilities = torch.softmax(logits, dim=1)

        # 获取预测结果
        pred_class = torch.argmax(logits, dim=1).item()
        pred_prob = probabilities[0].cpu().numpy()

        # 构建结果
        result = {
            'predicted_class': self.classes[pred_class],
            'predicted_class_idx': pred_class,
            'confidence': float(pred_prob[pred_class]),
            'probabilities': {
                self.classes[i]: float(prob)
                for i, prob in enumerate(pred_prob)
            },
            'image_path': image_path
        }

        return result

    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        批量预测

        Args:
            image_paths: 图像路径列表

        Returns:
            预测结果列表
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"处理 {image_path} 失败: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })

        return results

    def get_tumor_probability(self, image_path: str) -> Tuple[float, str]:
        """
        获取肿瘤概率（多分类版本）

        Args:
            image_path: 图像路径

        Returns:
            (肿瘤概率, 预测类别)
        """
        result = self.predict(image_path)

        # 计算肿瘤概率（所有肿瘤类别的概率之和，排除notumor）
        tumor_prob = sum([
            prob for class_name, prob in result['probabilities'].items()
            if class_name.lower() != 'notumor'
        ])

        return tumor_prob, result['predicted_class']

    def get_detailed_analysis(self, image_path: str) -> Dict:
        """
        获取详细分析结果（多分类版本）

        Args:
            image_path: 图像路径

        Returns:
            详细分析结果
        """
        result = self.predict(image_path)

        # 排序概率
        sorted_probs = sorted(
            result['probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 判断是否检测到肿瘤
        is_tumor = result['predicted_class'].lower() != 'notumor'

        # 计算肿瘤概率（所有肿瘤类别的概率之和，排除notumor）
        tumor_probability = sum([
            prob for class_name, prob in result['probabilities'].items()
            if class_name.lower() != 'notumor'
        ])

        analysis = {
            'image_path': image_path,
            'primary_prediction': result['predicted_class'],
            'confidence': result['confidence'],
            'all_predictions': [
                {
                    'class': class_name,
                    'probability': prob,
                    'percentage': f"{prob*100:.2f}%"
                }
                for class_name, prob in sorted_probs
            ],
            'is_tumor': is_tumor,
            'tumor_probability': tumor_probability,
            'tumor_type': result['predicted_class'] if is_tumor else 'none'
        }

        return analysis

    def generate_medical_text(self,
                             diagnosis: str,
                             confidence: float,
                             probabilities: Dict,
                             text_type: str = "imaging_findings") -> str:
        """
        使用MedGemma生成医学文本

        Args:
            diagnosis: 诊断类别
            confidence: 置信度
            probabilities: 各类别概率
            text_type: 文本类型 ('imaging_findings', 'clinical_advice', 'differential_diagnosis')

        Returns:
            生成的医学文本
        """
        try:
            # 构建医学提示词
            prompt = self._build_medical_prompt(diagnosis, confidence, probabilities, text_type)

            logger.info(f"使用模型生成{text_type}文本...")
            logger.debug(f"提示词: {prompt[:100]}...")

            # 准备输入
            inputs = self.processor(text=prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 生成文本
            with torch.no_grad():
                output_ids = self.full_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )

            # 解码生成的文本
            generated_text = self.processor.tokenizer.decode(
                output_ids[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()

            logger.info(f"✓ 文本生成成功，长度: {len(generated_text)}")
            return generated_text

        except Exception as e:
            logger.error(f"文本生成失败: {e}")
            return ""

    def _build_medical_prompt(self,
                              diagnosis: str,
                              confidence: float,
                              probabilities: Dict,
                              text_type: str) -> str:
        """
        构建医学提示词
        """
        diagnosis_display = diagnosis.upper() if diagnosis else "UNKNOWN"
        confidence_pct = f"{confidence*100:.1f}%"

        # 构建概率分布文本
        prob_text = "\n".join([
            f"• {class_name}: {prob*100:.1f}%"
            for class_name, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        ])

        if text_type == "imaging_findings":
            prompt = f"""Based on the MRI brain imaging analysis, the AI model has detected:

Diagnosis: {diagnosis_display}
Confidence: {confidence_pct}

Probability Distribution:
{prob_text}

Please provide detailed imaging findings (in Chinese) describing:
1. Signal characteristics in T1 and T2 weighted images
2. Enhancement patterns after contrast
3. Size and location of any lesions
4. Surrounding brain edema and compression
5. Relationship to nearby structures

Use professional radiology terminology and format similar to hospital diagnostic reports.

详细影像学所见（用中文，符合医院诊断报告风格）："""

        elif text_type == "clinical_advice":
            prompt = f"""Based on the brain MRI diagnosis:

Diagnosis: {diagnosis_display}
Confidence: {confidence_pct}

Please provide comprehensive clinical recommendations (in Chinese) including:
1. Recommended follow-up imaging
2. Suggested clinical departments for consultation
3. Treatment options and considerations
4. Monitoring and follow-up plans
5. Precautions and emergency signs

Use professional medical terminology and provide evidence-based recommendations.

临床医学建议（用中文，专业医学术语）："""

        elif text_type == "differential_diagnosis":
            prompt = f"""Based on the brain MRI analysis with {confidence_pct} confidence for {diagnosis_display}:

Probability Distribution:
{prob_text}

Please provide differential diagnosis analysis (in Chinese) discussing:
1. Primary diagnosis likelihood and supporting features
2. Differential diagnoses to consider
3. Distinguishing imaging features between diagnoses
4. Clinical history considerations
5. Additional tests that might help confirm diagnosis

Use professional neuroradiology terminology.

鉴别诊断分析（用中文，神经放射学专业术语）："""

        else:  # general_summary
            prompt = f"""Based on the brain MRI imaging analysis:

Diagnosis: {diagnosis_display}
Confidence: {confidence_pct}

Probability Distribution:
{prob_text}

Please provide a comprehensive summary (in Chinese) of:
1. Main findings and diagnosis
2. Key imaging characteristics
3. Clinical significance
4. Recommended actions
5. Important notes for clinicians

诊断总结（用中文，专业医学描述）："""

        return prompt

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 初始化模型
    model_path = "medgemma_tumor_classifier/best_model.pt"

    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        exit(1)

    model = TumorDetectionModel(model_path)

    # 示例：单张图像预测
    test_image = "test_image.jpg"  # 替换为实际图像路径

    if os.path.exists(test_image):
        logger.info(f"\n预测 {test_image}...")
        result = model.predict(test_image)
        logger.info(f"预测结果: {json.dumps(result, indent=2)}")

        # 获取详细分析
        analysis = model.get_detailed_analysis(test_image)
        logger.info(f"\n详细分析:\n{json.dumps(analysis, indent=2)}")

        # 获取肿瘤概率
        tumor_prob, pred_class = model.get_tumor_probability(test_image)
        logger.info(f"\n肿瘤概率: {tumor_prob:.4f}")
        logger.info(f"预测类别: {pred_class}")
    else:
        logger.warning(f"测试图像不存在: {test_image}")

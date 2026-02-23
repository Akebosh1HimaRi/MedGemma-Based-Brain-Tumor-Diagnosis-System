# 🧠 MedGuide 脑部肿瘤检测系统

## 项目目的

MedGuide 是一个基于 AI 的医学影像诊断辅助系统，用于帮助医学专业人士快速诊断脑部肿瘤。系统使用深度学习模型 MedGemma 1.5 4B，通过 LoRA 微调技术，实现对脑部 MRI 影像的四分类诊断（胶质瘤、脑膜瘤、垂体瘤、无肿瘤）。

**重要声明**：本系统仅供医学专业人士参考，最终诊断应由有资质的医生确认。

---

## 项目内容

### 系统架构

系统采用分层架构，实现高效的脑部肿瘤诊断：

```
用户上传图像
    ↓
网页界面 (HTML5/CSS3/JS)
    ↓
Flask 后端 API
    ↓
模型推理模块 (TumorDetectionModel)
    ↓
MedGemma 1.5 4B + LoRA 微调
    ↓
分类和分析
    ↓
报告生成模块 (医学报告)
    ↓
HTML/TXT 报告下载
```

**数据流**：
1. 用户通过网页界面上传 MRI 影像
2. Flask API 验证和处理图像
3. 模型进行特征提取和分类
4. 系统生成 4 个类别的概率分布
5. 自动医学评估（恶性程度、紧急程度、就诊科室）
6. 生成可下载的医学报告

### 核心功能

| 功能 | 说明 |
|------|------|
| **四分类诊断** | 识别四种脑部病变：胶质瘤、脑膜瘤、垂体瘤、无肿瘤 |
| **概率分布** | 显示所有类别的诊断概率 |
| **医学评估** | 自动评估恶性程度、紧急程度、推荐就诊科室 |
| **报告生成** | 自动生成 HTML 和 TXT 格式的医学诊断报告 |
| **Web 界面** | 支持拖拽上传、实时预览、在线诊断 |
| **API 接口** | RESTful API 支持第三方集成 |
| **双语支持** | 完整的英文和中文界面及报告 |
| **离线使用** | 本地模型加载，无需网络依赖 |

### 四分类诊断详解

系统诊断四种脑部病变，以下是详细说明：

#### 1. **胶质瘤 (Glioma)**
- **恶性程度**：高（置信度 > 0.8 时）
- **紧急程度**：高 - 需要立即干预
- **推荐科室**：神经外科 / 肿瘤科
- **医学特征**：
  - 信号不均匀，边界不清晰
  - 可见坏死囊变区，周围脑水肿明显
  - 侵润性生长特征
  - 增强后呈不均匀强化
- **临床意义**：最具侵袭性的肿瘤，需要紧急治疗

#### 2. **脑膜瘤 (Meningioma)**
- **恶性程度**：低-中等（置信度 > 0.8 时）
- **紧急程度**：中等 - 建议定期监测
- **推荐科室**：神经外科
- **医学特征**：
  - 信号相对均匀，边界清晰
  - 脑膜尾征明显（特征性表现）
  - 增强后呈均匀明显强化
  - 周围脑水肿程度轻-中等
- **临床意义**：通常为良性，无症状时可观察治疗

#### 3. **垂体瘤 (Pituitary Tumor)**
- **恶性程度**：中等（置信度 > 0.8 时）
- **紧急程度**：中等 - 需要内分泌评估
- **推荐科室**：神经外科 / 内分泌科
- **医学特征**：
  - 位于鞍区
  - 信号均匀或略不均匀
  - 可能压迫视交叉（引起视觉症状）
  - 增强后呈均匀强化
- **临床意义**：影响激素分泌，需要激素学评估

#### 4. **无肿瘤 (No Tumor / Normal)**
- **恶性程度**：无（正常）
- **紧急程度**：低
- **推荐科室**：神经内科（如有症状）
- **医学特征**：
  - 脑实质信号均匀
  - 脑室系统形态正常
  - 脑沟、脑池清晰
  - 脑中线结构无偏移
- **临床意义**：未检测到病变，定期复查

### 诊断输出示例

系统提供：
- **主要分类**：病变属于哪个类别
- **置信度评分**：模型的确定程度（0-100%）
- **概率分布**：所有 4 个类别的百分比
- **医学评估**：恶性程度、紧急程度、推荐科室
- **自动报告**：包含影像分析和建议的详细医学报告

---

- ⚡ **快速启动**：启动速度提升 30-60 倍（本地模型）
- 🎯 **高精度**：测试精度 90%+
- 📊 **完整功能**：从诊断到报告生成的一站式解决方案
- 🌐 **双语支持**：英文和中文完全本地化

---

## 环境配置

### 硬件要求

| 组件 | 要求 |
|------|------|
| GPU | NVIDIA RTX 3090 / RTX 4090（32GB+ 显存） |
| CPU | 16核+ 处理器 |
| 内存 | 32GB+ RAM |
| 存储 | 100GB+ （模型 + 数据） |

### 软件环境

- **OS**：Windows 10+、Ubuntu 20.04+、macOS 12+
- **Python**：3.10+
- **CUDA**：12.1+
- **cuDNN**：8.9+

### 安装步骤

```bash
# 1. 克隆或下载项目
cd med_guide

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载预训练模型（首次运行）
python download_model.py
# 选择下载方法 1（Hugging Face CLI），等待 5-30 分钟

# 4. 完成！
```

---

## 项目使用方法

### 方式 1：Web 界面（推荐）

```bash
# 启动 Web 应用
python app.py

# 访问网页
http://localhost:5000
```

**使用步骤：**
1. 拖拽或点击上传 MRI 脑部影像
2. 点击「开始分析」按钮
3. 查看诊断结果和概率分布
4. 点击「生成报告」获取医学诊断报告
5. 选择报告语言（英文/中文）并生成
6. 下载 HTML 或 TXT 报告

**支持格式**：PNG、JPG、JPEG、GIF、BMP、TIFF（最大 50MB）

### 方式 2：Python 脚本

```python
from inference import TumorDetectionModel
from report_generator import MedicalReportGenerator

# 加载模型
model = TumorDetectionModel('medgemma_tumor_classifier/best_model.pt')

# 预测单张图像
result = model.predict('path/to/image.jpg')
print(f"诊断: {result['predicted_class']}")
print(f"置信度: {result['confidence']:.2%}")

# 生成报告
generator = MedicalReportGenerator()
patient_info = {
    'patient_id': 'P001',
    'exam_date': '2026-02-19',
    'tumor_location': '右侧额叶'
}
text_report = generator.generate_text_report(
    result,
    patient_info,
    model=model,
    language='zh'  # 或 'en'
)
print(text_report)
```

### 方式 3：API 接口

```bash
# 预测
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/predict

# 生成报告
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "prediction": {...},
    "image_path": "uploads/...",
    "patient_info": {...},
    "report_language": "en"
  }' \
  http://localhost:5000/api/generate-report
```

---

## 训练方法

### 准备数据集

数据集目录结构：
```
dataset/
├── Train/
│   ├── glioma/      (3,773 张图像)
│   ├── meningioma/  (2,729 张图像)
│   ├── pituitary/   (3,130 张图像)
│   └── notumor/     (2,432 张图像)
└── Test/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── notumor/
```

### 配置训练参数

编辑 `train_model_lora.py` 中的 `Config` 类：

```python
class Config:
    # 模型
    use_local_model = True                  # 使用本地模型
    model_name = "models/medgemma-1.5-4b-it"

    # 训练参数
    num_epochs = 15                         # 训练轮数
    batch_size = 4                          # 批大小（根据 GPU 显存调整）
    learning_rate = 2e-4                    # 学习率

    # LoRA 参数
    lora_r = 16                             # LoRA 秩
    lora_alpha = 32                         # LoRA alpha
    lora_dropout = 0.05                     # dropout

    # 其他
    image_size = 224                        # 图像尺寸
    num_workers = 4                         # 数据加载线程
```

### 开始训练

```bash
# 训练模型
python train_model_lora.py

# 预期输出
# 加载数据集...
# 训练样本数: 9650
# 测试样本数: 2414
# 开始训练...
# Epoch 1/15
# 训练: 100%|████████████| 180/180 [12:34<00:00, 4.20s/it]
# 训练损失: 1.2345, 训练精度: 0.7823
# 测试损失: 0.9876, 测试精度: 0.8145
# ✓ 保存最佳模型
```

### 训练输出

```
medgemma_tumor_classifier/
├── best_model.pt              # 最佳模型（生产使用）
├── final_model.pt             # 最终模型
├── training_history.json      # 训练曲线数据
├── config.json                # 模型配置
└── checkpoints/               # 检查点文件
```

---

## 项目文件结构

```
med_guide/
├── app.py                          # Flask Web 应用主程序
├── inference.py                    # 模型推理模块
├── report_generator.py             # 医学报告生成模块
├── train_model_lora.py             # LoRA 微调训练脚本
├── download_model.py               # 模型下载脚本
├── requirements.txt                # 依赖列表
├── README.md                       # 英文说明文档
├── README_CN.md                    # 中文说明文档
│
├── templates/
│   └── index.html                  # Web 界面（HTML + CSS + JS）
│
├── models/
│   └── medgemma-1.5-4b-it/        # 预训练模型（下载后）
│
├── medgemma_tumor_classifier/     # 训练输出目录
│   ├── best_model.pt
│   ├── final_model.pt
│   ├── training_history.json
│   └── config.json
│
├── uploads/                        # 用户上传的图像
└── reports/                        # 生成的医学报告
```

### 关键文件说明

| 文件 | 功能 |
|------|------|
| `app.py` | Flask 后端服务，提供 Web 和 API 接口 |
| `inference.py` | 模型推理，图像预处理和预测 |
| `report_generator.py` | 生成 HTML/TXT 格式的医学报告 |
| `train_model_lora.py` | LoRA 微调训练脚本 |
| `templates/index.html` | 前端界面，支持英文/中文切换 |
| `download_model.py` | 从 Hugging Face 下载预训练模型 |

---

## 快速命令参考

```bash
# 首次使用（完整流程）
python download_model.py          # 下载模型（5-30 分钟）
python train_model_lora.py        # 微调模型（1-2 小时）
python app.py                     # 启动 Web 应用

# 仅运行推理（模型已训练）
python app.py

# 批量预测
python -c "from inference import TumorDetectionModel; \
  model = TumorDetectionModel('medgemma_tumor_classifier/best_model.pt'); \
  results = model.predict_batch(['img1.jpg', 'img2.jpg']); \
  print(results)"
```

---

## 性能指标

| 指标 | 值 |
|------|------|
| 启动时间 | 1-2 秒（本地模型） |
| 推理时间 | 1-2 秒/图像 |
| 训练精度 | 95%+ |
| 测试精度 | 90%+ |
| 内存占用 | ~20GB |
| 报告生成 | < 5 秒 |

---

## 快速故障排除指南

### 常见问题和解决方案

#### 1. **CUDA 相关错误**

**错误提示**：`RuntimeError: CUDA out of memory`

**解决方案**：
```python
# 方案 1：在 train_model_lora.py 中减少批大小
batch_size = 2  # 或 1

# 方案 2：使用梯度累积
gradient_accumulation_steps = 2

# 方案 3：使用混合精度训练
torch_dtype = torch.float16

# 方案 4：使用 CPU（速度较慢）
device = "cpu"
```

#### 2. **模型加载失败**

**错误提示**：`FileNotFoundError: 找不到模型文件`

**解决方案**：
```bash
# 确保已下载模型
python download_model.py

# 或手动检查模型目录
ls -la models/medgemma-1.5-4b-it/

# 验证模型文件完整性
ls -lh models/medgemma-1.5-4b-it/pytorch_model.bin
```

#### 3. **端口被占用**

**错误提示**：`Address already in use` 或 `端口 5000 已被使用`

**解决方案**：
```bash
# 方案 1：使用不同端口
# 编辑 app.py，修改最后一行：
app.run(host='0.0.0.0', port=5001)  # 使用 5001

# 方案 2：杀死占用端口的进程
# Linux/Mac:
lsof -i :5000
kill -9 <PID>

# Windows:
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

#### 4. **GPU 显存不足**

**错误提示**：`CUDA out of memory` 或 `OutOfMemoryError`

**解决方案**：
```python
# 降低模型精度
model = TumorDetectionModel(
    'best_model.pt',
    device='cuda'
)
# 使用 8 位量化
from bitsandbytes.optim import Adam8bit

# 或使用 CPU
model = TumorDetectionModel('best_model.pt', device='cpu')
```

#### 5. **模型下载超时**

**错误提示**：`Connection timeout` 或 `HTTPError: 429 Too Many Requests`

**解决方案**：
```bash
# 方案 1：重试并增加超时时间
HF_HUB_READ_TIMEOUT=60 python download_model.py

# 方案 2：使用 Hugging Face 镜像（国内推荐）
export HF_ENDPOINT=https://hf-mirror.com
python download_model.py

# 方案 3：手动下载
# 访问：https://huggingface.co/google/medgemma-1.5-4b-it
# 手动下载并放在 models/ 目录中
```

#### 6. **图像处理错误**

**错误提示**：`PIL.UnidentifiedImageError` 或 `无效的图像文件`

**解决方案**：
```python
# 确保图像是有效格式
from PIL import Image

try:
    img = Image.open('your_image.jpg')
    img.verify()  # 验证有效性
except Exception as e:
    print(f"图像错误: {e}")

# 支持格式：PNG、JPG、JPEG、GIF、BMP、TIFF
# 最大文件：50MB
```

#### 7. **Web 界面无响应**

**错误提示**：`Failed to fetch` 或 `连接被拒绝`

**解决方案**：
```bash
# 检查 Flask 是否运行
curl http://localhost:5000/api/health

# 检查日志中的错误
# 使用详细输出运行
python -u app.py

# 验证模型是否加载
curl http://localhost:5000/api/model-info
```

---

**Q：如何更改推理模型？**
```python
model = TumorDetectionModel(
    'medgemma_tumor_classifier/final_model.pt'  # 使用其他模型
)
```

**Q：GPU 显存不足怎么办？**
```python
# 在 train_model_lora.py 中修改
batch_size = 2  # 改为更小的值
```

**Q：如何生成中文报告？**
```python
# 在 Web 界面：选择报告生成 → 语言选择 → 中文
# 或在代码中：
generator.generate_text_report(result, patient_info, language='zh')
```

**Q：支持哪些图像格式？**
支持 PNG、JPG、JPEG、GIF、BMP、TIFF，最大 50MB

---

## 免责声明

⚠️ **重要提示**：
- 本系统仅供医学专业人士参考
- AI 诊断结果仅作辅助参考，最终诊断必须由有资质的医学专业人士确认
- 不建议用于独立的临床决策
- 使用本系统产生的结果由用户自行负责

---

## 许可证

本项目遵循 Apache 2.0 许可证。

---

## 版本信息

- **当前版本**：v2.0.0
- **更新日期**：2026 年 2 月
- **状态**：生产就绪

---

**祝你使用愉快！** 🚀

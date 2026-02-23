# 🧠 MedGuide Brain Tumor Detection System

## Project Purpose

MedGuide is an AI-powered medical imaging diagnostic assistance system designed to help medical professionals quickly diagnose brain tumors. The system uses the deep learning model MedGemma 1.5 4B, with LoRA fine-tuning technology, to perform four-class classification on brain MRI images (glioma, meningioma, pituitary tumor, no tumor).

**Important Notice**: This system is for medical professionals only. Final diagnosis must be confirmed by qualified physicians.

---

## Project Contents

### System Architecture

The system uses a layered architecture for efficient brain tumor diagnosis:

```
User Upload Image
        ↓
   Web Interface (HTML5/CSS3/JS)
        ↓
   Flask Backend API
        ↓
   Model Inference (TumorDetectionModel)
        ↓
   MedGemma 1.5 4B + LoRA Fine-tuning
        ↓
   Classification & Analysis
        ↓
   Report Generation (Medical Report)
        ↓
   HTML/TXT Report Download
```

**Data Flow**:
1. User uploads MRI image via web interface
2. Flask API validates and processes the image
3. Model performs feature extraction and classification
4. System generates probability distribution for 4 classes
5. Automatic medical assessment (malignancy, urgency, department)
6. Generate downloadable medical report

### Core Features

| Feature | Description |
|---------|-------------|
| **Four-Class Diagnosis** | Identifies four types of brain lesions: glioma, meningioma, pituitary tumor, no tumor |
| **Probability Distribution** | Displays diagnostic probabilities for all categories |
| **Medical Assessment** | Automatically assesses malignancy level, urgency level, and recommended departments |
| **Report Generation** | Automatically generates medical diagnostic reports in HTML and TXT formats |
| **Web Interface** | Supports drag-and-drop upload, real-time preview, and online diagnosis |
| **API Interface** | RESTful API for third-party integration |
| **Bilingual Support** | Complete English and Chinese interface and reports |
| **Offline Usage** | Local model loading with no internet dependency |

### Four-Class Diagnosis Explained

The system diagnoses four types of brain conditions. Here's a detailed breakdown:

#### 1. **Glioma (胶质瘤)**
- **Malignancy Level**: High (if confidence > 0.8)
- **Urgency Level**: High - requires immediate attention
- **Recommended Department**: Neurosurgery / Oncology
- **Medical Features**:
  - Heterogeneous signal with irregular boundaries
  - Visible necrotic areas and surrounding edema
  - Infiltrative growth characteristics
  - Non-uniform enhancement after contrast
- **Clinical Significance**: Most aggressive tumor type, requires urgent intervention

#### 2. **Meningioma (脑膜瘤)**
- **Malignancy Level**: Low-Moderate (if confidence > 0.8)
- **Urgency Level**: Moderate - monitoring recommended
- **Recommended Department**: Neurosurgery
- **Medical Features**:
  - Homogeneous signal with clear boundaries
  - Prominent dural tail sign (characteristic feature)
  - Uniform enhancement after contrast
  - Mild to moderate surrounding edema
- **Clinical Significance**: Usually benign, often suitable for observation if asymptomatic

#### 3. **Pituitary Tumor (垂体瘤)**
- **Malignancy Level**: Moderate (if confidence > 0.8)
- **Urgency Level**: Moderate - endocrine assessment needed
- **Recommended Department**: Neurosurgery / Endocrinology
- **Medical Features**:
  - Located in sella turcica region
  - Homogeneous or slightly heterogeneous signal
  - Can compress optic chiasm (visual symptoms)
  - Uniform enhancement pattern
- **Clinical Significance**: Affects hormone production, requires hormonal evaluation

#### 4. **No Tumor (无肿瘤 / Normal)**
- **Malignancy Level**: None (Normal)
- **Urgency Level**: Low
- **Recommended Department**: Neurology (if symptomatic)
- **Medical Features**:
  - Homogeneous brain parenchyma
  - Normal ventricular system
  - Clear sulci and cisterns
  - No midline shift
- **Clinical Significance**: No pathology detected, routine follow-up as needed

### Diagnosis Output Example

The system provides:
- **Primary Classification**: Which category the lesion belongs to
- **Confidence Score**: How certain the model is (0-100%)
- **Probability Distribution**: Percentage for all 4 categories
- **Medical Assessment**: Malignancy level, urgency, and recommended department
- **Automatic Report**: Detailed medical report with imaging analysis and recommendations

---

- ⚡ **Fast Startup**: 30-60x faster launch speed (with local model)
- 🎯 **High Accuracy**: 90%+ test accuracy
- 📊 **Complete Solution**: One-stop solution from diagnosis to report generation
- 🌐 **Bilingual Support**: Fully localized in English and Chinese

---

## Environment Configuration

### Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA RTX 3090 / RTX 4090 (32GB+ VRAM) |
| CPU | 16+ core processor |
| Memory | 32GB+ RAM |
| Storage | 100GB+ (model + data) |

### Software Environment

- **OS**: Windows 10+, Ubuntu 20.04+, macOS 12+
- **Python**: 3.10+
- **CUDA**: 12.1+
- **cuDNN**: 8.9+

### Installation Steps

```bash
# 1. Clone or download the project
cd med_guide

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download pre-trained model (first run only)
python download_model.py
# Select download method 1 (Hugging Face CLI), wait 5-30 minutes

# 4. Done!
```

---

## Project Usage

### Method 1: Web Interface (Recommended)

```bash
# Start the web application
python app.py

# Access the webpage
http://localhost:5000
```

**Usage Steps:**
1. Drag and drop or click to upload brain MRI image
2. Click the "Analyze" button
3. View diagnostic results and probability distribution
4. Click "Generate Report" to create a medical diagnostic report
5. Select report language (English/Chinese) and generate
6. Download the HTML or TXT report

**Supported Formats**: PNG, JPG, JPEG, GIF, BMP, TIFF (max 50MB)

### Method 2: Python Script

```python
from inference import TumorDetectionModel
from report_generator import MedicalReportGenerator

# Load model
model = TumorDetectionModel('medgemma_tumor_classifier/best_model.pt')

# Predict single image
result = model.predict('path/to/image.jpg')
print(f"Diagnosis: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")

# Generate report
generator = MedicalReportGenerator()
patient_info = {
    'patient_id': 'P001',
    'exam_date': '2026-02-19',
    'tumor_location': 'Right frontal lobe'
}
text_report = generator.generate_text_report(
    result,
    patient_info,
    model=model,
    language='en'  # or 'zh'
)
print(text_report)
```

### Method 3: API Interface

```bash
# Prediction
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/predict

# Generate report
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

## Training Method

### Prepare Dataset

Dataset directory structure:
```
dataset/
├── Train/
│   ├── glioma/      (3,773 images)
│   ├── meningioma/  (2,729 images)
│   ├── pituitary/   (3,130 images)
│   └── notumor/     (2,432 images)
└── Test/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── notumor/
```

### Configure Training Parameters

Edit the `Config` class in `train_model_lora.py`:

```python
class Config:
    # Model
    use_local_model = True                  # Use local model
    model_name = "models/medgemma-1.5-4b-it"

    # Training parameters
    num_epochs = 15                         # Training epochs
    batch_size = 4                          # Batch size (adjust based on GPU VRAM)
    learning_rate = 2e-4                    # Learning rate

    # LoRA parameters
    lora_r = 16                             # LoRA rank
    lora_alpha = 32                         # LoRA alpha
    lora_dropout = 0.05                     # Dropout

    # Others
    image_size = 224                        # Image size
    num_workers = 4                         # Data loading threads
```

### Start Training

```bash
# Train the model
python train_model_lora.py

# Expected output
# Loading dataset...
# Training samples: 9650
# Test samples: 2414
# Starting training...
# Epoch 1/15
# Training: 100%|████████████| 180/180 [12:34<00:00, 4.20s/it]
# Train loss: 1.2345, Train accuracy: 0.7823
# Test loss: 0.9876, Test accuracy: 0.8145
# ✓ Saving best model
```

### Training Output

```
medgemma_tumor_classifier/
├── best_model.pt              # Best model (for production)
├── final_model.pt             # Final model
├── training_history.json      # Training curve data
├── config.json                # Model configuration
└── checkpoints/               # Checkpoint files
```

---

## Project File Structure

```
med_guide/
├── app.py                          # Flask web application
├── inference.py                    # Model inference module
├── report_generator.py             # Medical report generation module
├── train_model_lora.py             # LoRA fine-tuning training script
├── download_model.py               # Model download script
├── requirements.txt                # Dependencies list
├── readme.md                       # English documentation
├── readme-cn.md                    # Chinese documentation
│
├── templates/
│   └── index.html                  # Web interface (HTML + CSS + JS)
│
├── models/
│   └── medgemma-1.5-4b-it/        # Pre-trained model (after download)
│
├── medgemma_tumor_classifier/     # Training output directory
│   ├── best_model.pt
│   ├── final_model.pt
│   ├── training_history.json
│   └── config.json
│
├── uploads/                        # User uploaded images
└── reports/                        # Generated medical reports
```

### Key File Descriptions

| File | Function |
|------|----------|
| `app.py` | Flask backend service, provides web and API interfaces |
| `inference.py` | Model inference, image preprocessing and prediction |
| `report_generator.py` | Generates medical reports in HTML/TXT formats |
| `train_model_lora.py` | LoRA fine-tuning training script |
| `templates/index.html` | Frontend interface, supports English/Chinese switching |
| `download_model.py` | Download pre-trained model from Hugging Face |

---

## Quick Command Reference

```bash
# First-time usage (complete workflow)
python download_model.py          # Download model (5-30 minutes)
python train_model_lora.py        # Fine-tune model (1-2 hours)
python app.py                     # Start web application

# Inference only (model already trained)
python app.py

# Batch prediction
python -c "from inference import TumorDetectionModel; \
  model = TumorDetectionModel('medgemma_tumor_classifier/best_model.pt'); \
  results = model.predict_batch(['img1.jpg', 'img2.jpg']); \
  print(results)"
```

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| Startup Time | 1-2 seconds (local model) |
| Inference Time | 1-2 seconds/image |
| Training Accuracy | 95%+ |
| Test Accuracy | 90%+ |
| Memory Usage | ~20GB |
| Report Generation | < 5 seconds |

---

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. **CUDA Related Errors**

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# Option 1: Reduce batch size in train_model_lora.py
batch_size = 2  # or 1 if still insufficient

# Option 2: Use gradient accumulation
gradient_accumulation_steps = 2

# Option 3: Use mixed precision training
torch_dtype = torch.float16

# Option 4: Use CPU (slower)
device = "cpu"
```

#### 2. **Model Loading Failed**

**Error**: `FileNotFoundError: Model file not found`

**Solutions**:
```bash
# Ensure you've downloaded the model first
python download_model.py

# Or manually check the model directory
ls -la models/medgemma-1.5-4b-it/

# Verify model file integrity
ls -lh models/medgemma-1.5-4b-it/pytorch_model.bin
```

#### 3. **Port Already in Use**

**Error**: `Address already in use` or `Port 5000 is already in use`

**Solutions**:
```bash
# Option 1: Use a different port
# Edit app.py, change the last line:
app.run(host='0.0.0.0', port=5001)  # Use 5001 instead

# Option 2: Kill the process using port 5000
# Linux/Mac:
lsof -i :5000
kill -9 <PID>

# Windows:
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

#### 4. **GPU VRAM Insufficient**

**Error**: `CUDA out of memory` or `OutOfMemoryError`

**Solutions**:
```python
# Reduce model precision
model = TumorDetectionModel(
    'best_model.pt',
    device='cuda'
)
# Use 8-bit quantization
from bitsandbytes.optim import Adam8bit

# Or use CPU
model = TumorDetectionModel('best_model.pt', device='cpu')
```

#### 5. **Model Download Timeout**

**Error**: `Connection timeout` or `HTTPError: 429 Too Many Requests`

**Solutions**:
```bash
# Option 1: Retry with increased timeout
HF_HUB_READ_TIMEOUT=60 python download_model.py

# Option 2: Use Hugging Face mirror (for China)
export HF_ENDPOINT=https://hf-mirror.com
python download_model.py

# Option 3: Manual download from Hugging Face
# Visit: https://huggingface.co/google/medgemma-1.5-4b-it
# Download manually and place in models/ directory
```

#### 6. **Image Processing Error**

**Error**: `PIL.UnidentifiedImageError` or `Invalid image file`

**Solutions**:
```python
# Ensure image is a valid format
from PIL import Image

try:
    img = Image.open('your_image.jpg')
    img.verify()  # Verify it's valid
except Exception as e:
    print(f"Image error: {e}")

# Supported formats: PNG, JPG, JPEG, GIF, BMP, TIFF
# Max file size: 50MB
```

#### 7. **Web Interface Not Responding**

**Error**: `Failed to fetch` or `Connection refused`

**Solutions**:
```bash
# Check if Flask is running
curl http://localhost:5000/api/health

# Check logs for errors
# Run with verbose output
python -u app.py

# Verify model is loaded
curl http://localhost:5000/api/model-info
```

---

**Q: How do I change the inference model?**
```python
model = TumorDetectionModel(
    'medgemma_tumor_classifier/final_model.pt'  # Use alternative model
)
```

**Q: What if GPU VRAM is insufficient?**
```python
# In train_model_lora.py, modify:
batch_size = 2  # Use smaller value
```

**Q: How do I generate reports in different languages?**
```python
# In web interface: Generate Report → Select Language → Choose
# Or in code:
generator.generate_text_report(result, patient_info, language='zh')
```

**Q: What image formats are supported?**
PNG, JPG, JPEG, GIF, BMP, TIFF (max 50MB)

---

## Disclaimer

⚠️ **Important Notice**:
- This system is for medical professionals only
- AI diagnostic results are for reference only; final diagnosis must be confirmed by qualified medical professionals
- Not recommended for independent clinical decision-making
- Users are responsible for any consequences resulting from using this system

---

## License

This project is licensed under the Apache 2.0 License.

---

## Version Information

- **Current Version**: v2.0.0
- **Last Updated**: February 2026
- **Status**: Production Ready

---

**Enjoy using MedGuide!** 🚀

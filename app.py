"""
脑部肿瘤检测Web应用
使用Flask框架提供Web界面
"""

import os
import torch
import json
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from datetime import datetime
import logging
from inference import TumorDetectionModel
from report_generator import MedicalReportGenerator
import io
from PIL import Image
import base64

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask应用配置
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['REPORTS_FOLDER'] = 'reports'

# 允许的文件类型
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# 全局模型实例
model = None
report_generator = None

def allowed_file(filename):
    """检查文件是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_model():
    """初始化模型"""
    global model, report_generator

    model_path = "medgemma_tumor_classifier/best_model.pt"

    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return False

    try:
        model = TumorDetectionModel(model_path)
        report_generator = MedicalReportGenerator()
        logger.info("模型初始化成功")
        return True
    except Exception as e:
        logger.error(f"模型初始化失败: {e}")
        return False

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """预测端点"""
    try:
        # 检查模型是否加载
        if model is None:
            return jsonify({'error': '模型未加载'}), 500

        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({'error': '未上传文件'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': '不支持的文件类型'}), 400

        # 保存上传的文件
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], timestamp + filename)
        file.save(filepath)

        logger.info(f"处理文件: {filepath}")

        # 进行预测
        result = model.predict(filepath)

        # 获取详细分析
        analysis = model.get_detailed_analysis(filepath)

        # 计算肿瘤概率（所有肿瘤类别的概率之和，排除notumor）
        tumor_prob = sum([
            prob for class_name, prob in result['probabilities'].items()
            if class_name.lower() != 'notumor'
        ])

        # 构建响应
        response = {
            'success': True,
            'prediction': {
                'class': result['predicted_class'],
                'confidence': result['confidence'],
                'tumor_probability': tumor_prob,
                'probabilities': result['probabilities']
            },
            'analysis': analysis,
            'image_path': filepath,
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"预测失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    """生成报告"""
    try:
        if model is None or report_generator is None:
            return jsonify({'error': '模型或报告生成器未加载'}), 500

        data = request.json

        # 获取预测结果
        prediction_result = data.get('prediction', {})
        image_path = data.get('image_path', '')
        patient_info = data.get('patient_info', {})
        report_language = data.get('report_language', 'en')  # 获取报告语言

        logger.info(f"开始生成报告（语言: {report_language}）...")
        logger.info("注意：文本生成需要额外处理时间，请耐心等待...")

        # 生成文本报告（传递model对象和report_language用于文本生成）
        text_report = report_generator.generate_text_report(
            prediction_result,
            patient_info,
            model=model,
            language=report_language
        )

        # 生成HTML报告（传递model对象和report_language用于文本生成）
        html_report = report_generator.generate_html_report(
            prediction_result,
            image_path=image_path,
            patient_info=patient_info,
            model=model,
            language=report_language
        )

        # 保存报告
        os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)
        report_id = patient_info.get('report_id', report_generator._generate_report_id())
        text_report_path = os.path.join(app.config['REPORTS_FOLDER'], f"{report_id}.txt")
        html_report_path = os.path.join(app.config['REPORTS_FOLDER'], f"{report_id}.html")

        report_generator.save_report(text_report, text_report_path, format='txt')
        report_generator.save_report(html_report, html_report_path, format='html')

        response = {
            'success': True,
            'report_id': report_id,
            'text_report': text_report,
            'html_report': html_report,
            'text_report_path': text_report_path,
            'html_report_path': html_report_path,
            'report_language': report_language,
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"生成报告失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/download-report/<report_id>/<format>', methods=['GET'])
def download_report(report_id, format):
    """下载报告"""
    try:
        if format not in ['txt', 'html']:
            return jsonify({'error': '不支持的格式'}), 400

        report_path = os.path.join(app.config['REPORTS_FOLDER'], f"{report_id}.{format}")

        if not os.path.exists(report_path):
            return jsonify({'error': '报告不存在'}), 404

        return send_file(
            report_path,
            as_attachment=True,
            download_name=f"report_{report_id}.{format}"
        )

    except Exception as e:
        logger.error(f"下载报告失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/image/<filename>', methods=['GET'])
def get_image(filename):
    """获取上传的图像"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if not os.path.exists(filepath):
            return jsonify({'error': '图像不存在'}), 404

        return send_file(filepath)

    except Exception as e:
        logger.error(f"获取图像失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/image-base64/<filename>', methods=['GET'])
def get_image_base64(filename):
    """获取图像的Base64编码"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if not os.path.exists(filepath):
            return jsonify({'error': '图像不存在'}), 404

        with open(filepath, 'rb') as f:
            image_base64 = base64.b64encode(f.read()).decode()

        return jsonify({
            'success': True,
            'image_base64': image_base64,
            'filename': filename
        })

    except Exception as e:
        logger.error(f"获取图像Base64失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """获取模型信息"""
    if model is None:
        return jsonify({'error': '模型未加载'}), 500

    return jsonify({
        'success': True,
        'model_name': 'MedGemma 1.5 4B',
        'fine_tuning_method': 'LoRA',
        'classes': model.classes,
        'num_classes': model.num_classes,
        'device': str(model.device)
    })

@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return jsonify({'error': '页面不存在'}), 404

@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    logger.error(f"内部错误: {error}")
    return jsonify({'error': '内部服务器错误'}), 500

if __name__ == '__main__':
    # 初始化模型
    if not init_model():
        logger.error("无法初始化模型，应用启动失败")
        exit(1)

    # 创建必要的文件夹
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)

    # 启动Flask应用
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )

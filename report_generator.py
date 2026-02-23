"""
医学报告生成模块
根据模型预测结果生成详细的医学诊断报告
"""

import os
from datetime import datetime
from typing import Dict, List
from PIL import Image
import base64
from io import BytesIO
import json

class MedicalReportGenerator:
    """医学报告生成器"""

    # 肿瘤类型描述 - 中文
    TUMOR_DESCRIPTIONS_ZH = {
        'tumor': '检测到脑部肿瘤病灶',
        'no_tumor': '未检测到明显的脑肿瘤病灶',
        'glioma': '胶质瘤是源于脑胶质细胞的恶性肿瘤，通常具有浸润性生长特征',
        'meningioma': '脑膜瘤是起源于硬膜的良性肿瘤，通常边界清楚',
        'pituitary': '垂体瘤是垂体腺的肿瘤，可引起内分泌异常'
    }

    # 肿瘤类型描述 - 英文
    TUMOR_DESCRIPTIONS_EN = {
        'tumor': 'Brain tumor lesion detected',
        'no_tumor': 'No obvious brain tumor lesion detected',
        'glioma': 'Glioma is a malignant tumor originating from glial cells with infiltrative growth characteristics',
        'meningioma': 'Meningioma is a benign tumor originating from the dura mater with clear boundaries',
        'pituitary': 'Pituitary tumor is an adenoma of the pituitary gland that can cause endocrine abnormalities'
    }

    # 影像学特征 - 中文
    IMAGING_FEATURES_ZH = {
        'tumor': '信号不均匀，可见异常占位性病灶，周围脑水肿明显',
        'no_tumor': '脑实质信号均匀，未见异常占位性病灶，脑室形态正常',
        'glioma': '信号不均匀，可见坏死囊变区，周围脑水肿明显，增强不均匀',
        'meningioma': '信号相对均匀，边界清晰，脑膜尾征明显，均匀强化',
        'pituitary': '位于鞍区，信号均匀，与垂体柄关系清楚'
    }

    # 影像学特征 - 英文
    IMAGING_FEATURES_EN = {
        'tumor': 'Heterogeneous signal with abnormal space-occupying lesion and significant surrounding brain edema',
        'no_tumor': 'Homogeneous brain parenchyma signal with no abnormal mass and normal ventricular morphology',
        'glioma': 'Heterogeneous signal with necrotic changes, significant surrounding edema, and heterogeneous enhancement',
        'meningioma': 'Relatively homogeneous signal with clear boundaries, prominent dural tail sign, and uniform enhancement',
        'pituitary': 'Located in sella turcica with homogeneous signal and clear relationship with pituitary stalk'
    }

    # 临床建议 - 中文
    CLINICAL_RECOMMENDATIONS_ZH = {
        'tumor': [
            '建议进行进一步的影像学检查（如CT、增强MRI）以明确诊断',
            '建议神经外科医生进行临床评估',
            '根据肿瘤类型和位置，制定个体化治疗方案',
            '定期随访监测肿瘤进展情况'
        ],
        'notumor': [
            '未见明显异常，建议定期复查',
            '如患者有临床症状，可考虑进一步神经学评估',
            '建议每年进行一次定期体检',
            '如症状持续，建议转诊神经内科进行进一步评估'
        ],
        'glioma': [
            '建议立即转诊神经外科进行进一步评估',
            '建议进行组织学活检以明确诊断和分级',
            '考虑进行分子病理检查（IDH、MGMT等）以指导治疗',
            '评估肿瘤可切除性，制定多学科治疗方案',
            '可考虑手术切除、放疗或化疗等综合治疗',
            '建议进行功能性MRI评估肿瘤与重要脑功能区的关系'
        ],
        'meningioma': [
            '建议定期影像学随访观察',
            '如肿瘤增大或出现症状，考虑手术切除',
            '必要时可进行放射治疗',
            '定期复查MRI（建议每3-6个月一次）',
            '评估肿瘤与重要血管和神经的关系',
            '如无症状且肿瘤稳定，可采用保守治疗策略'
        ],
        'pituitary': [
            '建议进行内分泌激素水平检查（包括GH、ACTH、TSH、LH、FSH等）',
            '若有症状，考虑药物治疗或手术切除',
            '定期影像学复查和内分泌评估',
            '必要时进行视野检查以评估视神经压迫情况',
            '建议转诊内分泌科进行激素水平评估和管理',
            '如有视觉症状，建议眼科会诊'
        ]
    }

    # 临床建议 - 英文
    CLINICAL_RECOMMENDATIONS_EN = {
        'tumor': [
            'Recommend further imaging studies (CT, enhanced MRI) for definitive diagnosis',
            'Clinical evaluation by neurosurgeon is advised',
            'Develop individualized treatment plan based on tumor type and location',
            'Regular follow-up to monitor tumor progression'
        ],
        'notumor': [
            'No obvious abnormality detected. Regular follow-up imaging is recommended',
            'If patient has clinical symptoms, consider further neurological evaluation',
            'Annual routine physical examination is advised',
            'If symptoms persist, refer to neurology for further assessment'
        ],
        'glioma': [
            'Immediate referral to neurosurgery for further evaluation',
            'Histopathological biopsy recommended to confirm diagnosis and grade',
            'Consider molecular pathology testing (IDH, MGMT, etc.) to guide treatment',
            'Assess tumor resectability and develop multidisciplinary treatment plan',
            'Consider multimodal treatment including surgery, radiotherapy, and chemotherapy',
            'Functional MRI recommended to assess relationship between tumor and eloquent brain areas'
        ],
        'meningioma': [
            'Regular imaging follow-up is recommended',
            'Consider surgery if tumor enlarges or symptoms develop',
            'Radiotherapy may be considered when necessary',
            'Regular MRI follow-up (every 3-6 months recommended)',
            'Assess relationship between tumor and vital vessels and nerves',
            'Conservative treatment strategy can be adopted if asymptomatic and tumor is stable'
        ],
        'pituitary': [
            'Endocrine hormone level testing recommended (GH, ACTH, TSH, LH, FSH, etc.)',
            'Consider medical or surgical treatment if symptomatic',
            'Regular imaging follow-up and endocrine evaluation',
            'Visual field testing may be necessary to assess optic nerve compression',
            'Refer to endocrinology for hormone level assessment and management',
            'Ophthalmology consultation recommended if visual symptoms present'
        ]
    }

    # 肿瘤位置描述
    TUMOR_LOCATIONS = {
        'frontal': '额叶',
        'temporal': '颞叶',
        'parietal': '顶叶',
        'occipital': '枕叶',
        'cerebellum': '小脑',
        'brainstem': '脑干',
        'ventricle': '脑室',
        'pineal': '松果体',
        'pituitary': '垂体',
        'unknown': '位置不确定'
    }

    def __init__(self):
        pass

    def _get_report_template(self, language: str = 'en') -> str:
        """获取报告模板（支持多语言）"""
        if language == 'zh':
            return self._get_report_template_zh()
        else:
            return self._get_report_template_en()

    def _get_report_template_en(self) -> str:
        """获取英文报告模板"""
        return """
╔════════════════════════════════════════════════════════════════════════════╗
║                      Brain MRI Imaging Diagnostic Report                   ║
║                    AI-Assisted Diagnostic System Report                     ║
╚════════════════════════════════════════════════════════════════════════════╝

【Report Information】
Report ID: {report_id}
Generation Time: {timestamp}
Examination Method: Brain MRI Scan with Contrast Enhancement
Analysis Method: Deep Learning AI-Assisted Diagnostic System
Analysis Model: MedGemma 1.5 4B (LoRA Fine-tuning)

【Patient Information】
Patient ID: {patient_id}
Examination Date: {exam_date}
Image ID: {image_id}

【AI Diagnostic Results】
Primary Diagnosis: {primary_diagnosis}
Diagnostic Confidence: {confidence_percentage}
Tumor Probability: {tumor_probability_percentage}

【Probability Distribution】
{probability_distribution}

【Detailed Imaging Findings】
{detailed_findings}

【Clinical Presentation】
{clinical_presentation}

【Imaging Features】
{imaging_features}

【Tumor Location Analysis】
{tumor_location}

【Diagnostic Conclusion】
Based on comprehensive MRI imaging features and AI model analysis, the diagnosis is: {diagnosis_conclusion}

This diagnosis is based on the AI deep learning model (MedGemma 1.5 4B LoRA fine-tuning) with a confidence level of {confidence_percentage}.
Diagnostic accuracy depends on image quality and the representativeness of the model's training data.

【Clinical Recommendations】
{clinical_recommendations}

【Risk Assessment】
Tumor Malignancy Level: {malignancy_assessment}
Urgency Level: {urgency_level}
Recommended Department: {recommended_department}

【Image Comparison】
{comparison_info}

【Quality Assessment】
Image Quality: {image_quality}
Diagnostic Confidence: {diagnostic_confidence}

【Disclaimer】
• This report is generated by an AI model for reference by medical professionals only
• Final diagnosis must be confirmed by qualified radiologists
• Recommend comprehensive analysis combining patient clinical history, physical examination, and other imaging modalities
• If there is any discrepancy or clinical presentation is inconsistent, further examination or specialist consultation is recommended
• This system cannot replace clinical judgment of physicians

【Technical Information】
Model Version: MedGemma 1.5 4B
Fine-tuning Method: LoRA (Low-Rank Adaptation)
Training Dataset: Brain MRI imaging dataset (~10,000 images)
Input Image Size: 224×224
Processing Time: {processing_time}ms

【Contact Information】
For questions or suggestions, please contact the Medical Imaging AI Team
Email: medguide@example.com
"""

    def _get_report_template_zh(self) -> str:
        """获取中文报告模板"""
        return """
╔════════════════════════════════════════════════════════════════════════════╗
║                         脑部MRI影像诊断报告                                  ║
║                    AI辅助诊断系统生成报告                                    ║
╚════════════════════════════════════════════════════════════════════════════╝

【报告信息】
报告编号: {report_id}
生成时间: {timestamp}
检查方式: 脑MRI平扫+增强扫描
分析方法: 深度学习AI辅助诊断系统
分析模型: MedGemma 1.5 4B (LoRA微调)

【患者信息】
患者ID: {patient_id}
检查日期: {exam_date}
图像编号: {image_id}

【AI诊断结果】
主要诊断: {primary_diagnosis}
诊断置信度: {confidence_percentage}
肿瘤概率: {tumor_probability_percentage}

【诊断概率分布】
{probability_distribution}

【详细影像学所见】
{detailed_findings}

【临床表现】
{clinical_presentation}

【影像学特征】
{imaging_features}

【肿瘤位置分析】
{tumor_location}

【诊断结论】
综合MRI影像特征和AI模型分析，诊断为：{diagnosis_conclusion}

本诊断基于AI深度学习模型（MedGemma 1.5 4B LoRA微调），置信度为{confidence_percentage}。
诊断准确性取决于图像质量和模型训练数据的代表性。

【临床建议】
{clinical_recommendations}

【风险评估】
肿瘤恶性程度评估: {malignancy_assessment}
紧急程度: {urgency_level}
建议就诊科室: {recommended_department}

【影像对比】
{comparison_info}

【质量评估】
图像质量: {image_quality}
诊断可信度: {diagnostic_confidence}

【免责声明】
• 本报告由AI模型生成，仅供医学专业人士参考
• 最终诊断应由有资质的放射科医生确认
• 建议结合患者临床病史、体格检查和其他影像学方法进行综合分析
• 如有异议或临床表现不符，建议进一步检查或寻求专家意见
• 本系统不能替代医生的临床判断

【技术信息】
模型版本: MedGemma 1.5 4B
微调方法: LoRA (Low-Rank Adaptation)
训练数据集: 脑部MRI影像数据集 (~10,000张)
输入图像尺寸: 224×224
处理时间: {processing_time}ms

【联系方式】
如有问题或建议，请联系医学影像AI团队
Email: medguide@example.com
"""

    def generate_text_report(self, prediction_result: Dict, patient_info: Dict = None, model=None, language: str = 'en') -> str:
        """
        生成文本格式的报告

        Args:
            prediction_result: 模型预测结果
            patient_info: 患者信息
            model: TumorDetectionModel 实例（用于生成医学文本）
            language: 报告语言 ('en' 或 'zh')

        Returns:
            报告文本
        """
        if patient_info is None:
            patient_info = {}

        # 选择语言特定的描述
        tumor_descriptions = self.TUMOR_DESCRIPTIONS_ZH if language == 'zh' else self.TUMOR_DESCRIPTIONS_EN
        imaging_features = self.IMAGING_FEATURES_ZH if language == 'zh' else self.IMAGING_FEATURES_EN
        clinical_recommendations = self.CLINICAL_RECOMMENDATIONS_ZH if language == 'zh' else self.CLINICAL_RECOMMENDATIONS_EN

        # 提取预测信息
        primary_diagnosis = prediction_result.get('predicted_class', 'Unknown')
        confidence = prediction_result.get('confidence', 0.0)
        probabilities = prediction_result.get('probabilities', {})

        # 如果诊断是Unknown或为空，从概率最高的类别推断
        if primary_diagnosis == 'Unknown' or primary_diagnosis is None or primary_diagnosis == '':
            if probabilities:
                primary_diagnosis = max(probabilities.items(), key=lambda x: x[1])[0]
            else:
                primary_diagnosis = 'notumor'

        # 计算肿瘤概率（多分类版本）
        # 计算肿瘤概率（所有肿瘤类别的概率之和，排除notumor）
        tumor_prob = sum([
            prob for class_name, prob in probabilities.items()
            if class_name.lower() != 'notumor'
        ])

        # 生成概率分布
        prob_dist = self._generate_probability_distribution(probabilities)

        # 获取诊断信息
        clinical_presentation = tumor_descriptions.get(
            primary_diagnosis.lower(),
            tumor_descriptions.get('tumor', 'Abnormal lesion detected' if language == 'en' else '检测到异常病灶')
        )

        imaging_feature = imaging_features.get(
            primary_diagnosis.lower(),
            imaging_features.get('tumor', 'Abnormal signal' if language == 'en' else '信号异常')
        )

        # 生成详细的影像所见部分
        # 优先使用模型生成的文本，如果模型不可用则使用预定义的文本
        if model is not None:
            try:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"使用模型生成影像学所见（语言: {language}）...")
                detailed_findings = model.generate_medical_text(
                    primary_diagnosis,
                    confidence,
                    probabilities,
                    text_type="imaging_findings"
                )
                if not detailed_findings:
                    logger.warning("模型生成失败，使用预定义文本")
                    detailed_findings = self._generate_detailed_findings(primary_diagnosis, confidence, probabilities, language)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"使用模型生成影像学所见失败: {e}，使用预定义文本")
                detailed_findings = self._generate_detailed_findings(primary_diagnosis, confidence, probabilities, language)
        else:
            detailed_findings = self._generate_detailed_findings(primary_diagnosis, confidence, probabilities, language)

        # 获取临床建议
        # 优先使用模型生成的文本
        if model is not None:
            try:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"使用模型生成临床建议（语言: {language}）...")
                clinical_advice_text = model.generate_medical_text(
                    primary_diagnosis,
                    confidence,
                    probabilities,
                    text_type="clinical_advice"
                )
                if clinical_advice_text:
                    recommendations_text = clinical_advice_text
                else:
                    logger.warning("模型生成失败，使用预定义建议")
                    recommendations = clinical_recommendations.get(
                        primary_diagnosis.lower(),
                        clinical_recommendations.get('tumor', [])
                    )
                    bullet_char = '•' if language == 'zh' else '•'
                    recommendations_text = '\n'.join([f"{bullet_char} {rec}" for rec in recommendations])
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"使用模型生成临床建议失败: {e}，使用预定义建议")
                recommendations = clinical_recommendations.get(
                    primary_diagnosis.lower(),
                    clinical_recommendations.get('tumor', [])
                )
                bullet_char = '•' if language == 'zh' else '•'
                recommendations_text = '\n'.join([f"{bullet_char} {rec}" for rec in recommendations])
        else:
            # 获取临床建议
            recommendations = clinical_recommendations.get(
                primary_diagnosis.lower(),
                clinical_recommendations.get('tumor', [])
            )
            bullet_char = '•' if language == 'zh' else '•'
            recommendations_text = '\n'.join([f"{bullet_char} {rec}" for rec in recommendations])

        # 根据语言选择报告模板
        report_template = self._get_report_template(language)

        # 生成报告
        report = report_template.format(
            report_id=patient_info.get('report_id', self._generate_report_id()),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            patient_id=patient_info.get('patient_id', 'N/A'),
            exam_date=patient_info.get('exam_date', datetime.now().strftime("%Y-%m-%d")),
            image_id=patient_info.get('image_id', 'N/A'),
            primary_diagnosis=primary_diagnosis,
            confidence_percentage=f"{confidence*100:.2f}%",
            tumor_probability_percentage=f"{tumor_prob*100:.2f}%",
            probability_distribution=prob_dist,
            detailed_findings=detailed_findings,
            clinical_presentation=clinical_presentation,
            imaging_features=imaging_feature,
            tumor_location=patient_info.get('tumor_location', '位置不确定' if language == 'zh' else 'Location uncertain'),
            diagnosis_conclusion=primary_diagnosis,
            clinical_recommendations=recommendations_text,
            malignancy_assessment=self._assess_malignancy(primary_diagnosis, confidence, language),
            urgency_level=self._assess_urgency(primary_diagnosis, confidence, language),
            recommended_department=self._get_recommended_department(primary_diagnosis, language),
            comparison_info=patient_info.get('comparison_info', '无对比检查' if language == 'zh' else 'No comparison study'),
            image_quality=patient_info.get('image_quality', '良好' if language == 'zh' else 'Good'),
            diagnostic_confidence=self._get_diagnostic_confidence(confidence, language),
            processing_time=patient_info.get('processing_time', 'N/A')
        )

        return report

    def generate_html_report(self, prediction_result: Dict, image_path: str = None,
                            patient_info: Dict = None, model=None, language: str = 'en') -> str:
        """
        生成HTML格式的报告

        Args:
            prediction_result: 模型预测结果
            image_path: 图像路径
            patient_info: 患者信息
            model: TumorDetectionModel 实例（用于生成医学文本）

        Returns:
            HTML报告
        """
        if patient_info is None:
            patient_info = {}

        # 提取预测信息
        primary_diagnosis = prediction_result.get('predicted_class', 'Unknown')
        confidence = prediction_result.get('confidence', 0.0)
        probabilities = prediction_result.get('probabilities', {})

        # 如果诊断是Unknown或为空，从概率最高的类别推断
        if primary_diagnosis == 'Unknown' or primary_diagnosis is None or primary_diagnosis == '':
            if probabilities:
                primary_diagnosis = max(probabilities.items(), key=lambda x: x[1])[0]
            else:
                primary_diagnosis = 'notumor'

        # 计算肿瘤概率（多分类版本）
        # 计算肿瘤概率（所有肿瘤类别的概率之和，排除notumor）
        tumor_prob = sum([
            prob for class_name, prob in probabilities.items()
            if class_name.lower() != 'notumor'
        ])

        # 生成概率分布HTML
        prob_dist_html = self._generate_probability_distribution_html(probabilities)

        # 生成详细的影像所见部分
        # 优先使用模型生成的文本，如果模型不可用则使用预定义的文本
        use_predefined_findings = False
        if model is not None:
            try:
                import logging
                logger = logging.getLogger(__name__)
                logger.info("使用模型生成HTML影像学所见...")
                detailed_findings = model.generate_medical_text(
                    primary_diagnosis,
                    confidence,
                    probabilities,
                    text_type="imaging_findings"
                )
                if not detailed_findings:
                    logger.warning("模型生成失败，使用预定义文本")
                    detailed_findings = self._generate_detailed_findings(primary_diagnosis, confidence, probabilities, language)
                    use_predefined_findings = True
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"使用模型生成影像学所见失败: {e}，使用预定义文本")
                detailed_findings = self._generate_detailed_findings(primary_diagnosis, confidence, probabilities, language)
                use_predefined_findings = True
        else:
            detailed_findings = self._generate_detailed_findings(primary_diagnosis, confidence, probabilities, language)
            use_predefined_findings = True

        # 如果使用了预定义的findings，则转换为HTML格式（标题加粗，冒号后换行）
        if use_predefined_findings:
            detailed_findings_html = self._generate_detailed_findings_html()
        else:
            # 模型生成的文本，保持原样但做基本的HTML处理
            detailed_findings_html = detailed_findings.replace('\n', '<br/>\n')

        # 获取诊断信息（根据语言选择）
        tumor_descriptions = self.TUMOR_DESCRIPTIONS_ZH if language == 'zh' else self.TUMOR_DESCRIPTIONS_EN
        imaging_features_dict = self.IMAGING_FEATURES_ZH if language == 'zh' else self.IMAGING_FEATURES_EN
        clinical_recommendations = self.CLINICAL_RECOMMENDATIONS_ZH if language == 'zh' else self.CLINICAL_RECOMMENDATIONS_EN

        clinical_presentation = tumor_descriptions.get(
            primary_diagnosis.lower(),
            tumor_descriptions.get('tumor', 'Abnormal lesion detected' if language == 'en' else '检测到异常病灶')
        )

        imaging_features = imaging_features_dict.get(
            primary_diagnosis.lower(),
            imaging_features_dict.get('tumor', 'Abnormal signal' if language == 'en' else '信号异常')
        )

        # 获取临床建议
        # 优先使用模型生成的文本
        if model is not None:
            try:
                import logging
                logger = logging.getLogger(__name__)
                logger.info("使用模型生成HTML临床建议...")
                clinical_advice_text = model.generate_medical_text(
                    primary_diagnosis,
                    confidence,
                    probabilities,
                    text_type="clinical_advice"
                )
                if clinical_advice_text:
                    # 将生成的文本转换为HTML列表格式
                    recommendations_html = ''.join([
                        f"<li>{line.strip()}</li>"
                        for line in clinical_advice_text.split('\n')
                        if line.strip() and not line.strip().startswith('•')
                    ])
                    # 如果转换失败，使用预定义建议
                    if not recommendations_html:
                        recommendations = clinical_recommendations.get(
                            primary_diagnosis.lower(),
                            clinical_recommendations.get('tumor', [])
                        )
                        recommendations_html = ''.join([
                            f"<li>{rec}</li>" for rec in recommendations
                        ])
                else:
                    logger.warning("模型生成失败，使用预定义建议")
                    recommendations = clinical_recommendations.get(
                        primary_diagnosis.lower(),
                        clinical_recommendations.get('tumor', [])
                    )
                    recommendations_html = ''.join([
                        f"<li>{rec}</li>" for rec in recommendations
                    ])
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"使用模型生成临床建议失败: {e}，使用预定义建议")
                recommendations = clinical_recommendations.get(
                    primary_diagnosis.lower(),
                    clinical_recommendations.get('tumor', [])
                )
                recommendations_html = ''.join([
                    f"<li>{rec}</li>" for rec in recommendations
                ])
        else:
            recommendations = clinical_recommendations.get(
                primary_diagnosis.lower(),
                clinical_recommendations.get('tumor', [])
            )
            recommendations_html = ''.join([
                f"<li>{rec}</li>" for rec in recommendations
            ])

        # 转换图像为base64
        image_base64 = ""
        if image_path and os.path.exists(image_path):
            try:
                with open(image_path, 'rb') as f:
                    image_base64 = base64.b64encode(f.read()).decode()
            except Exception as e:
                print(f"无法读取图像: {e}")

        # 生成HTML
        html_lang = "zh-CN" if language == 'zh' else "en"
        html_title = "脑部MRI影像诊断报告" if language == 'zh' else "Brain MRI Imaging Diagnostic Report"
        html_header_title = "脑部MRI影像诊断报告" if language == 'zh' else "Brain MRI Imaging Diagnostic Report"
        html_header_subtitle = "AI辅助诊断系统生成报告" if language == 'zh' else "AI-Assisted Diagnostic System Report"

        html = f"""
<!DOCTYPE html>
<html lang="{html_lang}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html_title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            background-color: #f5f5f5;
            padding: 20px;
        }}

        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        .header {{
            text-align: center;
            border-bottom: 3px solid #2c3e50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}

        .header h1 {{
            color: #2c3e50;
            font-size: 28px;
            margin-bottom: 10px;
        }}

        .header p {{
            color: #7f8c8d;
            font-size: 14px;
        }}

        .section {{
            margin-bottom: 30px;
        }}

        .section-title {{
            background-color: #34495e;
            color: white;
            padding: 12px 15px;
            border-radius: 4px;
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 15px;
        }}

        .section-content {{
            padding: 15px;
            background-color: #ecf0f1;
            border-radius: 4px;
            line-height: 1.8;
        }}

        .info-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}

        .info-item {{
            background-color: white;
            padding: 15px;
            border-left: 4px solid #3498db;
            border-radius: 4px;
        }}

        .info-label {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}

        .info-value {{
            color: #34495e;
            font-size: 16px;
        }}

        .diagnosis-box {{
            background-color: #fff3cd;
            border: 2px solid #ffc107;
            border-radius: 4px;
            padding: 20px;
            margin-bottom: 20px;
        }}

        .diagnosis-box.warning {{
            background-color: #f8d7da;
            border-color: #f5c6cb;
        }}

        .diagnosis-box.success {{
            background-color: #d4edda;
            border-color: #c3e6cb;
        }}

        .diagnosis-title {{
            font-weight: bold;
            font-size: 18px;
            margin-bottom: 10px;
        }}

        .probability-chart {{
            margin: 20px 0;
        }}

        .probability-item {{
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }}

        .probability-label {{
            width: 120px;
            font-weight: bold;
            color: #2c3e50;
        }}

        .probability-bar {{
            flex: 1;
            height: 30px;
            background-color: #ecf0f1;
            border-radius: 4px;
            overflow: hidden;
            margin: 0 15px;
        }}

        .probability-fill {{
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2980b9);
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 10px;
            color: white;
            font-weight: bold;
            font-size: 12px;
        }}

        .probability-value {{
            width: 60px;
            text-align: right;
            font-weight: bold;
            color: #2c3e50;
        }}

        .image-container {{
            text-align: center;
            margin: 20px 0;
        }}

        .image-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}

        .recommendations {{
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            border-radius: 4px;
        }}

        .recommendations ul {{
            list-style-position: inside;
            line-height: 1.8;
        }}

        .recommendations li {{
            margin-bottom: 10px;
            color: #2c3e50;
        }}

        .footer {{
            border-top: 1px solid #bdc3c7;
            padding-top: 20px;
            margin-top: 30px;
            font-size: 12px;
            color: #7f8c8d;
            text-align: center;
        }}

        .disclaimer {{
            background-color: #fff3cd;
            border: 1px solid #ffc107;
            padding: 15px;
            border-radius: 4px;
            margin-top: 20px;
            font-size: 12px;
            color: #856404;
        }}

        .disclaimer ul {{
            list-style-position: inside;
            margin-top: 10px;
        }}

        .disclaimer li {{
            margin-bottom: 5px;
        }}

        @media print {{
            body {{
                background-color: white;
                padding: 0;
            }}

            .container {{
                box-shadow: none;
                padding: 0;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{html_header_title}</h1>
            <p>{html_header_subtitle}</p>
        </div>

        <div class="section">
            <div class="section-title">{'报告信息' if language == 'zh' else 'Report Information'}</div>
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">{'报告编号' if language == 'zh' else 'Report ID'}</div>
                    <div class="info-value">{patient_info.get('report_id', self._generate_report_id())}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">{'生成时间' if language == 'zh' else 'Generation Time'}</div>
                    <div class="info-value">{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">{'患者ID' if language == 'zh' else 'Patient ID'}</div>
                    <div class="info-value">{patient_info.get('patient_id', 'N/A')}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">{'检查日期' if language == 'zh' else 'Examination Date'}</div>
                    <div class="info-value">{patient_info.get('exam_date', datetime.now().strftime("%Y-%m-%d"))}</div>
                </div>
            </div>
        </div>

        <div class="section">
            <div class="section-title">{'AI诊断结果' if language == 'zh' else 'AI Diagnostic Results'}</div>
            <div class="diagnosis-box {'warning' if tumor_prob > 0.5 else 'success'}">
                <div class="diagnosis-title">{'主要诊断: ' if language == 'zh' else 'Primary Diagnosis: '}{primary_diagnosis}</div>
                <div style="margin-top: 10px;">
                    <strong>{'诊断置信度: ' if language == 'zh' else 'Diagnostic Confidence: '}</strong> {confidence*100:.2f}%<br>
                    <strong>{'肿瘤概率: ' if language == 'zh' else 'Tumor Probability: '}</strong> {tumor_prob*100:.2f}%
                </div>
            </div>

            <div class="probability-chart">
                <h4 style="margin-bottom: 15px;">{'诊断概率分布' if language == 'zh' else 'Probability Distribution'}</h4>
                {prob_dist_html}
            </div>
        </div>

        {f'<div class="section"><div class="section-title">{"检查图像" if language == "zh" else "Examination Image"}</div><div class="image-container"><img src="data:image/jpeg;base64,{image_base64}" alt="{"检查图像" if language == "zh" else "examination image"}"></div></div>' if image_base64 else ''}

        <div class="section">
            <div class="section-title">{'详细影像学所见' if language == 'zh' else 'Detailed Imaging Findings'}</div>
            <div class="section-content">
                <p style="line-height: 1.8;">{detailed_findings_html}</p>
            </div>
        </div>

        <div class="section">
            <div class="section-title">{'临床信息' if language == 'zh' else 'Clinical Information'}</div>
            <div class="section-content">
                <h4 style="margin-bottom: 10px;">{'临床表现' if language == 'zh' else 'Clinical Presentation'}</h4>
                <p>{clinical_presentation}</p>

                <h4 style="margin-top: 15px; margin-bottom: 10px;">{'影像学特征' if language == 'zh' else 'Imaging Features'}</h4>
                <p>{imaging_features}</p>
            </div>
        </div>

        <div class="section">
            <div class="section-title">{'临床建议' if language == 'zh' else 'Clinical Recommendations'}</div>
            <div class="recommendations">
                <ul>
                    {recommendations_html}
                </ul>
            </div>
        </div>

        <div class="section">
            <div class="section-title">{'风险评估' if language == 'zh' else 'Risk Assessment'}</div>
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">{'恶性程度' if language == 'zh' else 'Malignancy Level'}</div>
                    <div class="info-value">{self._assess_malignancy(primary_diagnosis, confidence, language)}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">{'紧急程度' if language == 'zh' else 'Urgency Level'}</div>
                    <div class="info-value">{self._assess_urgency(primary_diagnosis, confidence, language)}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">{'推荐科室' if language == 'zh' else 'Recommended Department'}</div>
                    <div class="info-value">{self._get_recommended_department(primary_diagnosis, language)}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">{'诊断可信度' if language == 'zh' else 'Diagnostic Confidence'}</div>
                    <div class="info-value">{self._get_diagnostic_confidence(confidence, language)}</div>
                </div>
            </div>
        </div>

        <div class="disclaimer">
            <strong>{'免责声明' if language == 'zh' else 'Disclaimer'}</strong>
            <ul>
                {'<li>本报告由AI模型生成，仅供医学专业人士参考</li><li>最终诊断应由有资质的放射科医生确认</li><li>建议结合患者临床病史、体格检查和其他影像学方法进行综合分析</li><li>如有异议或临床表现不符，建议进一步检查或寻求专家意见</li><li>本系统不能替代医生的临床判断</li>' if language == 'zh' else '<li>This report is generated by an AI model for reference by medical professionals only</li><li>Final diagnosis must be confirmed by qualified radiologists</li><li>Recommend comprehensive analysis combining patient clinical history, physical examination, and other imaging modalities</li><li>If there is any discrepancy or clinical presentation is inconsistent, further examination or specialist consultation is recommended</li><li>This system cannot replace clinical judgment of physicians</li>'}
            </ul>
        </div>

        <div class="footer">
            <p>{'模型版本: MedGemma 1.5 4B | 微调方法: LoRA | 生成时间: ' if language == 'zh' else 'Model Version: MedGemma 1.5 4B | Fine-tuning Method: LoRA | Generation Time: '}{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
    </div>
</body>
</html>
"""
        return html

    def _generate_detailed_findings(self, diagnosis: str, confidence: float, probabilities: Dict, language: str = 'en') -> str:
        """
        生成详细的影像学所见（支持多语言）
        返回格式：标题；内容1；内容2；...
        """
        diagnosis_lower = diagnosis.lower()
        findings_data = []  # 存储 (标题, 内容列表) 的元组

        if language == 'zh':
            # 脑实质基本评估
            brain_findings = []
            if diagnosis_lower == 'notumor':
                brain_findings = [
                    "脑实质信号均匀，未见明显占位性病灶",
                    "脑脊液信号正常，脑室系统大小形态正常",
                    "脑沟、脑池展开良好，未见异常压迹",
                    "脑中线结构无偏移"
                ]
            else:
                brain_findings = [
                    "脑实质内见异常信号病灶",
                    "脑脊液信号可见受压征象"
                ]
            findings_data.append(("脑实质及脑脊液信号分布", brain_findings))

            if diagnosis_lower == 'glioma':
                glioma_findings = [
                    "病灶信号不均匀，T1加权像呈等-略低信号",
                    "T2加权像呈高信号，FLAIR序列信号高",
                    "病灶边界欠清晰，侵润性生长特征明显",
                    "可见坏死囊变区，周围脑组织水肿明显",
                    "增强后病灶呈不均匀强化，中央可见坏死区"
                ]
                if confidence > 0.7:
                    glioma_findings.append("AI诊断置信度较高，恶性程度可能较高")
                findings_data.append(("胶质瘤特征性表现", glioma_findings))

            elif diagnosis_lower == 'meningioma':
                meningioma_findings = [
                    "病灶信号相对均匀，T1加权像呈等-略高信号",
                    "T2加权像呈等-略高信号，FLAIR序列信号高",
                    "病灶边界清晰，有明显的分界线",
                    "可见脑膜尾征（dural tail sign），具有特征性意义",
                    "增强后病灶呈均匀明显强化",
                    "周围脑组织水肿程度轻-中等",
                    "一般预后较好，多为良性肿瘤"
                ]
                findings_data.append(("脑膜瘤特征性表现", meningioma_findings))

            elif diagnosis_lower == 'pituitary':
                pituitary_findings = [
                    "病灶位于鞍区，与垂体柄关系密切",
                    "信号均匀，T1加权像呈等信号",
                    "T2加权像可呈等-高信号",
                    "增强后呈均匀强化",
                    "上方可见视交叉压迫，视神经受压",
                    "两侧海绵窦可见受压",
                    "建议进行内分泌激素水平检查"
                ]
                findings_data.append(("垂体肿瘤特征性表现", pituitary_findings))

            other_findings = [
                "小脑半球及脑干未见异常信号",
                "脑底池及侧脑室形态正常",
                "颅骨及头皮软组织未见明显异常"
            ]
            findings_data.append(("其他结构评估", other_findings))
        else:  # English
            brain_findings = []
            if diagnosis_lower == 'notumor':
                brain_findings = [
                    "Homogeneous brain parenchyma signal with no obvious space-occupying lesion",
                    "Normal cerebrospinal fluid signal and ventricular system size and morphology",
                    "Normal sulci and cisterns, no abnormal pressure impressions",
                    "No midline shift"
                ]
            else:
                brain_findings = [
                    "Abnormal signal lesion detected in brain parenchyma",
                    "Signs of compression visible in cerebrospinal fluid signal"
                ]
            findings_data.append(("Brain parenchyma and CSF signal distribution", brain_findings))

            if diagnosis_lower == 'glioma':
                glioma_findings = [
                    "Heterogeneous lesion signal, T1-weighted images show iso- to hypointense signal",
                    "T2-weighted images show hyperintense signal, high FLAIR signal",
                    "Poorly defined lesion boundaries with prominent infiltrative growth characteristics",
                    "Necrotic cystic changes visible with significant surrounding brain edema",
                    "Heterogeneous enhancement with central necrosis after contrast administration"
                ]
                if confidence > 0.7:
                    glioma_findings.append("High AI diagnostic confidence, tumor may be of higher malignancy grade")
                findings_data.append(("Characteristic glioma features", glioma_findings))

            elif diagnosis_lower == 'meningioma':
                meningioma_findings = [
                    "Relatively homogeneous lesion signal, T1-weighted images show iso- to slightly hyperintense signal",
                    "T2-weighted images show iso- to slightly hyperintense signal, high FLAIR signal",
                    "Well-defined lesion boundaries with clear demarcation",
                    "Prominent dural tail sign (dural tail sign) - characteristic finding",
                    "Uniform marked enhancement after contrast administration",
                    "Mild to moderate surrounding brain edema",
                    "Generally favorable prognosis, usually benign tumor"
                ]
                findings_data.append(("Characteristic meningioma features", meningioma_findings))

            elif diagnosis_lower == 'pituitary':
                pituitary_findings = [
                    "Lesion located in sella turcica with close relationship to pituitary stalk",
                    "Homogeneous signal, T1-weighted images show isointense signal",
                    "T2-weighted images may show iso- to hyperintense signal",
                    "Uniform enhancement after contrast administration",
                    "Optic chiasm compression visible superiorly, optic nerve compression",
                    "Bilateral cavernous sinus compression visible",
                    "Endocrine hormone level testing recommended"
                ]
                findings_data.append(("Characteristic pituitary tumor features", pituitary_findings))

            other_findings = [
                "No abnormal signal in cerebellar hemispheres and brainstem",
                "Normal cerebral basal cisterns and lateral ventricle morphology",
                "No obvious abnormality in skull and scalp soft tissues"
            ]
            findings_data.append(("Other structural assessment", other_findings))

        # 存储为实例变量供HTML方法使用
        self._findings_data = findings_data

        # 返回纯文本格式（用于文本报告）
        result_lines = []
        for title, contents in findings_data:
            separator = "；" if language == 'zh' else "; "
            result_lines.append(f"{title}：" + separator.join(contents))
        return "\n\n".join(result_lines)

    def _generate_detailed_findings_html(self) -> str:
        """
        生成HTML格式的详细影像学所见
        标题加粗，冒号后换行
        """
        if not hasattr(self, '_findings_data'):
            return ""

        html_parts = []
        for title, contents in self._findings_data:
            html_part = f"<strong>{title}：</strong><br/>\n"
            html_part += "；".join(contents)
            html_parts.append(html_part)

        return "<br/><br/>\n".join(html_parts)

    def _generate_probability_distribution(self, probabilities: Dict) -> str:
        """生成概率分布文本"""
        lines = []
        for class_name, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            bar_length = int(prob * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            lines.append(f"  {class_name:15} [{bar}] {prob*100:6.2f}%")
        return "\n".join(lines)

    def _generate_probability_distribution_html(self, probabilities: Dict) -> str:
        """生成概率分布HTML"""
        html_items = []
        for class_name, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            html_items.append(f"""
                <div class="probability-item">
                    <div class="probability-label">{class_name}</div>
                    <div class="probability-bar">
                        <div class="probability-fill" style="width: {prob*100}%">
                            {prob*100:.1f}%
                        </div>
                    </div>
                    <div class="probability-value">{prob*100:.2f}%</div>
                </div>
            """)
        return "".join(html_items)

    def _assess_malignancy(self, diagnosis: str, confidence: float, language: str = 'en') -> str:
        """评估恶性程度（多分类版本，支持多语言）"""
        diagnosis_lower = diagnosis.lower()

        if language == 'zh':
            if diagnosis_lower == 'notumor':
                return "无 (正常)"
            elif diagnosis_lower == 'glioma':
                if confidence > 0.8:
                    return "高"
                elif confidence > 0.6:
                    return "中-高"
                else:
                    return "中"
            elif diagnosis_lower == 'meningioma':
                if confidence > 0.8:
                    return "低-中"
                else:
                    return "低"
            elif diagnosis_lower == 'pituitary':
                if confidence > 0.8:
                    return "中"
                else:
                    return "低-中"
            else:
                return "未知"
        else:  # English
            if diagnosis_lower == 'notumor':
                return "None (Normal)"
            elif diagnosis_lower == 'glioma':
                if confidence > 0.8:
                    return "High"
                elif confidence > 0.6:
                    return "Moderate-High"
                else:
                    return "Moderate"
            elif diagnosis_lower == 'meningioma':
                if confidence > 0.8:
                    return "Low-Moderate"
                else:
                    return "Low"
            elif diagnosis_lower == 'pituitary':
                if confidence > 0.8:
                    return "Moderate"
                else:
                    return "Low-Moderate"
            else:
                return "Unknown"

    def _assess_urgency(self, diagnosis: str, confidence: float, language: str = 'en') -> str:
        """评估紧急程度（多分类版本，支持多语言）"""
        diagnosis_lower = diagnosis.lower()

        if language == 'zh':
            if diagnosis_lower == 'notumor':
                return "低"
            elif diagnosis_lower == 'glioma':
                if confidence > 0.8:
                    return "高"
                elif confidence > 0.6:
                    return "中"
                else:
                    return "中-低"
            elif diagnosis_lower == 'meningioma':
                if confidence > 0.8:
                    return "中"
                else:
                    return "低-中"
            elif diagnosis_lower == 'pituitary':
                if confidence > 0.8:
                    return "中"
                else:
                    return "低-中"
            else:
                return "未知"
        else:  # English
            if diagnosis_lower == 'notumor':
                return "Low"
            elif diagnosis_lower == 'glioma':
                if confidence > 0.8:
                    return "High"
                elif confidence > 0.6:
                    return "Moderate"
                else:
                    return "Moderate-Low"
            elif diagnosis_lower == 'meningioma':
                if confidence > 0.8:
                    return "Moderate"
                else:
                    return "Low-Moderate"
            elif diagnosis_lower == 'pituitary':
                if confidence > 0.8:
                    return "Moderate"
                else:
                    return "Low-Moderate"
            else:
                return "Unknown"

    def _get_recommended_department(self, diagnosis: str, language: str = 'en') -> str:
        """获取推荐科室（多分类版本，支持多语言）"""
        diagnosis_lower = diagnosis.lower()

        if language == 'zh':
            if diagnosis_lower == 'notumor':
                return "神经内科"
            elif diagnosis_lower == 'glioma':
                return "神经外科 / 肿瘤科"
            elif diagnosis_lower == 'meningioma':
                return "神经外科"
            elif diagnosis_lower == 'pituitary':
                return "神经外科 / 内分泌科"
            else:
                return "神经外科"
        else:  # English
            if diagnosis_lower == 'notumor':
                return "Neurology"
            elif diagnosis_lower == 'glioma':
                return "Neurosurgery / Oncology"
            elif diagnosis_lower == 'meningioma':
                return "Neurosurgery"
            elif diagnosis_lower == 'pituitary':
                return "Neurosurgery / Endocrinology"
            else:
                return "Neurosurgery"

    def _get_diagnostic_confidence(self, confidence: float, language: str = 'en') -> str:
        """获取诊断可信度（支持多语言）"""
        if language == 'zh':
            if confidence > 0.9:
                return "非常高"
            elif confidence > 0.8:
                return "高"
            elif confidence > 0.7:
                return "中等"
            elif confidence > 0.6:
                return "中等偏低"
            else:
                return "低"
        else:  # English
            if confidence > 0.9:
                return "Very High"
            elif confidence > 0.8:
                return "High"
            elif confidence > 0.7:
                return "Moderate"
            elif confidence > 0.6:
                return "Moderate-Low"
            else:
                return "Low"

    def _generate_report_id(self) -> str:
        """生成报告ID"""
        import uuid
        return f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{str(uuid.uuid4())[:8].upper()}"

    def save_report(self, report_content: str, output_path: str, format: str = 'txt'):
        """
        保存报告

        Args:
            report_content: 报告内容
            output_path: 输出路径
            format: 格式 ('txt' 或 'html')
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"报告已保存到: {output_path}")

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 示例预测结果
    prediction_result = {
        'predicted_class': 'tumor',
        'confidence': 0.85,
        'probabilities': {
            'tumor': 0.85,
            'no_tumor': 0.15
        }
    }

    # 示例患者信息
    patient_info = {
        'patient_id': 'P20240207001',
        'exam_date': '2024-02-07',
        'image_id': 'IMG001',
        'tumor_location': '右侧额叶',
        'image_quality': '良好',
        'processing_time': '1250'
    }

    # 生成报告
    generator = MedicalReportGenerator()

    # 生成文本报告
    text_report = generator.generate_text_report(prediction_result, patient_info)
    print(text_report)

    # 生成HTML报告
    html_report = generator.generate_html_report(prediction_result, patient_info=patient_info)

    # 保存报告
    generator.save_report(text_report, 'reports/report_20240207.txt', format='txt')
    generator.save_report(html_report, 'reports/report_20240207.html', format='html')

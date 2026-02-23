"""
多分类系统代码验证脚本
验证代码逻辑而不需要实际的模型和依赖
"""

import os
import sys
import io

# 设置输出编码
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_inference_logic():
    """测试推理逻辑"""
    print("\n" + "="*60)
    print("测试1：推理逻辑验证")
    print("="*60)

    # 模拟 get_tumor_probability 的逻辑
    mock_result = {
        'predicted_class': 'glioma',
        'confidence': 0.85,
        'probabilities': {
            'glioma': 0.50,
            'meningioma': 0.20,
            'pituitary': 0.15,
            'notumor': 0.15
        }
    }

    # 新的肿瘤概率计算逻辑
    tumor_classes = ['glioma', 'meningioma', 'pituitary']
    tumor_prob = max([
        prob for class_name, prob in mock_result['probabilities'].items()
        if class_name.lower() in tumor_classes
    ], default=0.0)

    expected_tumor_prob = 0.50
    if abs(tumor_prob - expected_tumor_prob) < 0.001:
        print(f"✓ 肿瘤概率计算正确: {tumor_prob:.4f}")
    else:
        print(f"❌ 肿瘤概率计算错误: {tumor_prob:.4f} (期望: {expected_tumor_prob})")
        return False

    # 测试 is_tumor 逻辑
    is_tumor = mock_result['predicted_class'].lower() != 'notumor'
    if is_tumor:
        print(f"✓ 肿瘤检测逻辑正确: is_tumor = {is_tumor}")
    else:
        print(f"❌ 肿瘤检测逻辑错误")
        return False

    # 测试 notumor 情况
    mock_result_notumor = {
        'predicted_class': 'notumor',
        'confidence': 0.90,
        'probabilities': {
            'glioma': 0.05,
            'meningioma': 0.02,
            'pituitary': 0.03,
            'notumor': 0.90
        }
    }

    is_tumor_notumor = mock_result_notumor['predicted_class'].lower() != 'notumor'
    if not is_tumor_notumor:
        print(f"✓ 无肿瘤检测逻辑正确: is_tumor = {is_tumor_notumor}")
    else:
        print(f"❌ 无肿瘤检测逻辑错误")
        return False

    return True

def test_report_logic():
    """测试报告生成逻辑"""
    print("\n" + "="*60)
    print("测试2：报告生成逻辑验证")
    print("="*60)

    # 模拟恶性程度评估逻辑
    def assess_malignancy(diagnosis, confidence):
        diagnosis_lower = diagnosis.lower()
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

    # 模拟紧急程度评估逻辑
    def assess_urgency(diagnosis, confidence):
        diagnosis_lower = diagnosis.lower()
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

    # 模拟科室推荐逻辑
    def get_recommended_department(diagnosis):
        diagnosis_lower = diagnosis.lower()
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

    # 测试用例
    test_cases = [
        ('glioma', 0.85, '高', '高', '神经外科 / 肿瘤科'),
        ('meningioma', 0.85, '低-中', '中', '神经外科'),
        ('pituitary', 0.85, '中', '中', '神经外科 / 内分泌科'),
        ('notumor', 0.90, '无 (正常)', '低', '神经内科'),
    ]

    all_passed = True
    for class_name, confidence, exp_mal, exp_urg, exp_dept in test_cases:
        mal = assess_malignancy(class_name, confidence)
        urg = assess_urgency(class_name, confidence)
        dept = get_recommended_department(class_name)

        if mal == exp_mal and urg == exp_urg and dept == exp_dept:
            print(f"✓ {class_name}: 恶性={mal}, 紧急={urg}, 科室={dept}")
        else:
            print(f"❌ {class_name} 逻辑错误")
            if mal != exp_mal:
                print(f"   恶性程度: {mal} (期望: {exp_mal})")
            if urg != exp_urg:
                print(f"   紧急程度: {urg} (期望: {exp_urg})")
            if dept != exp_dept:
                print(f"   推荐科室: {dept} (期望: {exp_dept})")
            all_passed = False

    return all_passed

def test_code_modifications():
    """测试代码修改是否正确"""
    print("\n" + "="*60)
    print("测试3：代码修改验证")
    print("="*60)

    base_dir = "D:\\Learning\\Code\\Vscode\\Projects\\Python\\Medguide\\med_guide"

    # 检查 inference.py 修改
    print("\n检查 inference.py...")
    inference_file = os.path.join(base_dir, "inference.py")
    if os.path.exists(inference_file):
        with open(inference_file, 'r', encoding='utf-8') as f:
            content = f.read()

        checks = [
            ("tumor_classes = ['glioma', 'meningioma', 'pituitary']", "肿瘤类别定义"),
            ("result['predicted_class'].lower() != 'notumor'", "无肿瘤判断逻辑"),
        ]

        for check_str, desc in checks:
            if check_str in content:
                print(f"✓ {desc} 已修改")
            else:
                print(f"❌ {desc} 未找到")
                return False
    else:
        print(f"❌ inference.py 不存在")
        return False

    # 检查 app.py 修改
    print("\n检查 app.py...")
    app_file = os.path.join(base_dir, "app.py")
    if os.path.exists(app_file):
        with open(app_file, 'r', encoding='utf-8') as f:
            content = f.read()

        if "tumor_classes = ['glioma', 'meningioma', 'pituitary']" in content:
            print(f"✓ 肿瘤概率计算已修改")
        else:
            print(f"❌ 肿瘤概率计算未修改")
            return False
    else:
        print(f"❌ app.py 不存在")
        return False

    # 检查 report_generator.py 修改
    print("\n检查 report_generator.py...")
    report_file = os.path.join(base_dir, "report_generator.py")
    if os.path.exists(report_file):
        with open(report_file, 'r', encoding='utf-8') as f:
            content = f.read()

        checks = [
            ("def _assess_malignancy(self, diagnosis: str, confidence: float) -> str:", "恶性程度方法"),
            ("def _assess_urgency(self, diagnosis: str, confidence: float) -> str:", "紧急程度方法"),
            ("def _get_recommended_department(self, diagnosis: str) -> str:", "科室推荐方法"),
            ("'notumor':", "无肿瘤类别"),
            ("'glioma':", "胶质瘤类别"),
            ("'meningioma':", "脑膜瘤类别"),
            ("'pituitary':", "垂体瘤类别"),
        ]

        for check_str, desc in checks:
            if check_str in content:
                print(f"✓ {desc} 已修改")
            else:
                print(f"❌ {desc} 未找到")
                return False
    else:
        print(f"❌ report_generator.py 不存在")
        return False

    # 检查 train_model_lora.py 修改
    print("\n检查 train_model_lora.py...")
    train_file = os.path.join(base_dir, "train_model_lora.py")
    if os.path.exists(train_file):
        with open(train_file, 'r', encoding='utf-8') as f:
            content = f.read()

        if "WeightedRandomSampler" in content:
            print(f"✓ 类权重处理已添加")
        else:
            print(f"❌ 类权重处理未添加")
            return False
    else:
        print(f"❌ train_model_lora.py 不存在")
        return False

    return True

def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("脑肿瘤检测系统 - 多分类代码验证")
    print("="*60)

    results = {
        '推理逻辑': test_inference_logic(),
        '报告生成逻辑': test_report_logic(),
        '代码修改': test_code_modifications(),
    }

    print("\n" + "="*60)
    print("验证总结")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ 通过" if result else "❌ 失败"
        print(f"{status}: {test_name}")

    print(f"\n总体: {passed}/{total} 验证通过")

    if passed == total:
        print("\n✓ 所有代码修改验证通过！")
        print("\n下一步:")
        print("1. 运行 python train_model_lora.py 进行模型训练")
        print("2. 运行 python app.py 启动 Web 应用")
        print("3. 在浏览器中访问 http://localhost:5000")
        return 0
    else:
        print(f"\n❌ 有 {total - passed} 个验证失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())

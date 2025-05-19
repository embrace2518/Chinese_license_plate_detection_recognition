# -*- coding: utf-8 -*-
import re
import matplotlib.pyplot as plt


def parse_log(log_path):
    epochs = []
    metrics = {
        'box': [], 'obj': [], 'cls': [], 'landmark': [], 'total': [],
        'P': [], 'R': [], 'mAP50': [], 'mAP5095': []
    }

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # 修正训练指标解析
            if line.startswith('Epoch'):
                # 匹配更精确的数字格式（包含小数点和科学计数法）
                parts = re.findall(r'\d+\.?\d*|\.?\d+', line)
                if len(parts) >= 11:  # 调整字段数量
                    try:
                        epochs.append(int(parts[0].split('/')[0]))
                        # 调整字段索引（跳过GPU内存部分）
                        metrics['box'].append(float(parts[4]))   # 原parts[2] → 4
                        metrics['obj'].append(float(parts[5]))   # 原parts[3] → 5
                        metrics['cls'].append(float(parts[6]))    # 原parts[4] → 6
                        metrics['landmark'].append(float(parts[7])) # 原parts[5] →7
                        metrics['total'].append(float(parts[8]))  # 原parts[6] →8
                    except:
                        continue

            # 修正验证指标解析
            elif 'all' in line and 'mAP@.5' in line:
                # 匹配更精确的浮点数格式
                parts = re.findall(r'\d+\.\d+', line)
                if len(parts) >= 4:
                    metrics['P'].append(float(parts[0]))
                    metrics['R'].append(float(parts[1]))
                    metrics['mAP50'].append(float(parts[2]))
                    metrics['mAP5095'].append(float(parts[3]))

    return epochs, metrics

def plot_training_curves(epochs, metrics):
    plt.figure(figsize=(14, 10))

    # 损失曲线
    plt.subplot(2, 1, 1)
    for key in ['box', 'obj', 'cls', 'landmark', 'total']:
        plt.plot(epochs, metrics[key], label=key)
    plt.title('Training Loss')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # 评估指标
    plt.subplot(2, 1, 2)
    for key in ['P', 'R', 'mAP50', 'mAP5095']:
        plt.plot(metrics[key], label=key)
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_analysis.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    log_path = "训练结果1.txt"  # 修正文件名乱码
    epochs, metrics = parse_log(log_path)
    if epochs:
        plot_training_curves(epochs, metrics)
    else:
        print("未解析到有效数据，请检查日志格式是否匹配")


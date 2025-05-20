# coding=gbk
import openpyxl
import json
import os

# ��ȡ Excel �ļ�
wb = openpyxl.load_workbook("CLPD.xlsx")
sheet = wb.active

# ָ�� JSON �ļ�����Ŀ¼
save_dir = r"/datasets/labels/train"
os.makedirs(save_dir, exist_ok=True)  # ȷ��Ŀ¼����

# ��������
for idx, row in enumerate(sheet.iter_rows(min_row=2, values_only=True)):  # ����������
    path, x1, y1, x2, y2, x3, y3, x4, y4, label = row  # ����ÿ���� 4 ����������

    # ���� JSON ���ݸ�ʽ
    json_data = {
        "shapes": [
            {
                "label": "single",
                "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            }
        ]
    }

    # ���� JSON �ļ�
    filename = os.path.join(save_dir, f"{idx}.json")  # ��������·��
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(json_data, f)
    if idx == 120:
        break

print("ת����ɣ�")

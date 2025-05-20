# coding=gbk
import openpyxl
import json
import os

# 读取 Excel 文件
wb = openpyxl.load_workbook("CLPD.xlsx")
sheet = wb.active

# 指定 JSON 文件保存目录
save_dir = r"/datasets/labels/train"
os.makedirs(save_dir, exist_ok=True)  # 确保目录存在

# 处理数据
for idx, row in enumerate(sheet.iter_rows(min_row=2, values_only=True)):  # 跳过标题行
    path, x1, y1, x2, y2, x3, y3, x4, y4, label = row  # 假设每行有 4 个坐标数据

    # 构造 JSON 数据格式
    json_data = {
        "shapes": [
            {
                "label": "single",
                "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            }
        ]
    }

    # 生成 JSON 文件
    filename = os.path.join(save_dir, f"{idx}.json")  # 生成完整路径
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(json_data, f)
    if idx == 120:
        break

print("转换完成！")

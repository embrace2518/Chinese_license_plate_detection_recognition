# coding=gbk
import openpyxl
import json
import os

base_dir = "D:/datasets/labels"
dirs = ['train', 'val', 'test']
[os.makedirs(os.path.join(base_dir, d), exist_ok=True) for d in dirs]
src_dir = "D:/datasets/images/det_result"


def get_json():
    wb = openpyxl.load_workbook("D:/datasets/CLPD.xlsx")
    sheet = wb.active
    # 获取用户输入
    train_rows = int(input("请输入训练集行数："))
    val_rows = int(input("请输入验证集行数："))
    test_rows = int(input("请输入测试集行数："))
    current_row = 2  # 从第二行开始
    for dir_name, max_rows in zip(dirs, [train_rows, val_rows, test_rows]):
        end_row = current_row + max_rows
        for idx, row in enumerate(sheet.iter_rows(min_row=current_row, max_row=end_row - 1, values_only=True)):
            path, x1, y1, x2, y2, x3, y3, x4, y4, label = row
            json_data = {
                "shapes": [
                    {
                        "label": "single",
                        "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    }
                ]
            }
            save_path = os.path.join(base_dir, dir_name)
            filename = os.path.join(save_path, f"{current_row - 2}.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(json_data, f)
            current_row += 1


def get_name_label():
    wb = openpyxl.load_workbook("D:/datasets/CLPD.xlsx")
    sheet = wb.active
    labels = {}
    for row in sheet.iter_rows(min_row=2, values_only=True):
        filename = row[0].split('/')[1]
        if filename and row[9]:
            labels[filename] = row[9]
    return labels


if __name__ == '__main__':
    choice = input("请选择模式 (1-分数据集生成JSON / 2-重命名图片): ")
    if choice == '1':
        get_json()
    elif choice == '2':
        labels = get_name_label()
        for _, filename in enumerate(os.listdir(src_dir)):
            if labels[filename]:
                src = os.path.join(src_dir, filename)
                new_name = f"{labels[filename]}_{filename}"
                dst = os.path.join(src_dir, new_name)
                os.rename(src, dst)
                print(f"重命名: {filename} -> {new_name}")

    else:
        print("无效的选项")
    print("操作完成！")

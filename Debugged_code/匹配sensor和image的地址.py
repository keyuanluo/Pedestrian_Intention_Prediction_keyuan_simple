import os
import re
import pandas as pd
from datetime import datetime
from math import inf

def parse_sensor_name(folder_name):
    """
    输入: "WEAR_28_01_2022_12_39_34"
    输出: datetime(2022,1,28,12,39,34)
    如果解析失败返回 None
    """
    try:
        # 去掉前缀 "WEAR_"
        # 剩余 "28_01_2022_12_39_34"
        parts = folder_name.replace("WEAR_", "").split("_")
        day   = int(parts[0])
        month = int(parts[1])
        year  = int(parts[2])
        hour  = int(parts[3])
        minute= int(parts[4])
        sec   = int(parts[5])
        return datetime(year, month, day, hour, minute, sec)
    except:
        return None

def parse_image_name(folder_name):
    """
    输入: "01_06_2022_18_24_49_scene1_stop_all"
    输出: datetime(2022,6,1,18,24,49)
    如果解析失败返回 None
    """
    try:
        parts = folder_name.split("_")
        day   = int(parts[0])
        month = int(parts[1])
        year  = int(parts[2])
        hour  = int(parts[3])
        minute= int(parts[4])
        sec   = int(parts[5])
        return datetime(year, month, day, hour, minute, sec)
    except:
        return None

def main():
    # 1) 指定根目录
    sensor_root = "/media/robert/4TB-SSD/watchped_dataset/Sensor"
    images_root = "/media/robert/4TB-SSD/watchped_dataset/Images"

    # 2) 扫描 sensor 目录并解析
    sensor_info = []
    for fname in os.listdir(sensor_root):
        fpath = os.path.join(sensor_root, fname)
        if os.path.isdir(fpath) and fname.startswith("WEAR_"):
            dt_sen = parse_sensor_name(fname)
            if dt_sen is not None:
                sensor_info.append({
                    "sensor_folder": fname,
                    "sensor_path": fpath,
                    "datetime": dt_sen
                })

    # 按时间排序（可选）
    sensor_info.sort(key=lambda x: x["datetime"])

    # 3) 扫描 images 目录(包括子文件夹、子子文件夹)
    image_info = []
    for subfolder in os.listdir(images_root):
        sub_path = os.path.join(images_root, subfolder)
        if os.path.isdir(sub_path):
            # 遍历子子文件夹
            for subsub in os.listdir(sub_path):
                subsub_path = os.path.join(sub_path, subsub)
                if os.path.isdir(subsub_path):
                    dt_img = parse_image_name(subsub)
                    if dt_img is not None:
                        image_info.append({
                            "image_folder": subsub,
                            "image_path": subsub_path,
                            "datetime": dt_img
                        })

    image_info.sort(key=lambda x: x["datetime"])

    # 4) 对每个 image 文件夹，找到时间最近的 sensor 文件夹
    results = []
    for img_item in image_info:
        img_dt = img_item["datetime"]

        best_sensor = None
        min_diff = inf

        for sen_item in sensor_info:
            diff_sec = abs((sen_item["datetime"] - img_dt).total_seconds())
            if diff_sec < min_diff:
                min_diff = diff_sec
                best_sensor = sen_item

        if best_sensor is not None:
            results.append({
                "image_folder": img_item["image_folder"],
                "image_path": img_item["image_path"],
                "sensor_folder": best_sensor["sensor_folder"],
                "sensor_path": best_sensor["sensor_path"],
                "time_diff_sec": min_diff
            })
        else:
            # 如果没有找到任何匹配，就跳过或做个标记
            results.append({
                "image_folder": img_item["image_folder"],
                "image_path": img_item["image_path"],
                "sensor_folder": "",
                "sensor_path": "",
                "time_diff_sec": None
            })

    # 5) 保存到 Excel
    df = pd.DataFrame(results)
    output_excel = "/media/robert/4TB-SSD/watchped_dataset/matched_folders.xlsx"
    df.to_excel(output_excel, index=False)
    print(f"匹配完成！结果已写入 {output_excel}")

if __name__ == "__main__":
    main()

import os
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

def parse_folder_name(folder_name, date_order):
    """
    根据 date_order 判断解析顺序:
      - "月日": folder_name 是 "06_01_2023_17_07_54_..." => 月_日_年_时_分_秒
      - "日月": folder_name 是 "01_06_2023_17_07_54_..." => 日_月_年_时_分_秒
    返回 datetime(...) 或 None
    """
    try:
        parts = folder_name.split("_")
        if date_order == "月日":
            # 月_日_年_时_分_秒
            month = int(parts[0])
            day   = int(parts[1])
            year  = int(parts[2])
            hour  = int(parts[3])
            minute= int(parts[4])
            sec   = int(parts[5])
        elif date_order == "日月":
            # 日_月_年_时_分_秒
            day   = int(parts[0])
            month = int(parts[1])
            year  = int(parts[2])
            hour  = int(parts[3])
            minute= int(parts[4])
            sec   = int(parts[5])
        else:
            return None  # 不支持的 date_order

        return datetime(year, month, day, hour, minute, sec)
    except:
        return None

def main():
    # 1) 读取 merged_result.xlsx
    merged_excel = "/media/robert/4TB-SSD/watchped_dataset/merged_result.xlsx"
    df_merged = pd.read_excel(merged_excel)
    # 假设里面包含: [video_id, original_path, width, height, date_order]

    # 2) 读取 Sensor 目录并解析
    sensor_root = "/media/robert/4TB-SSD/watchped_dataset/Sensor"
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
    # 排序(可选)
    sensor_info.sort(key=lambda x: x["datetime"])

    # 3) 从 merged_result 逐行取出 original_path, date_order, 并解析时间
    #    需要获取最后一级文件夹名
    results = []
    for idx, row in df_merged.iterrows():
        video_id     = row["video_id"]
        original_pth = row["original_path"]
        date_order   = row["date_order"]

        # last folder name
        # original_pth 可能是 ".../01_06_2023_17_07_54_scene7_stop_cross_veryfar"
        folder_name = os.path.basename(original_pth.rstrip("/"))  # 去掉末尾 /, 取最后一级

        dt_img = parse_folder_name(folder_name, date_order)
        if dt_img is None:
            # 解析失败, 也可记录
            results.append({
                "video_id": video_id,
                "original_path": original_pth,
                "sensor_folder": "",
                "sensor_path": "",
                "time_diff_sec": None
            })
            continue

        # 4) 匹配时间最近的 sensor
        best_sensor = None
        min_diff = inf
        for s in sensor_info:
            diff_sec = abs((s["datetime"] - dt_img).total_seconds())
            if diff_sec < min_diff:
                min_diff = diff_sec
                best_sensor = s

        if best_sensor:
            results.append({
                "video_id": video_id,
                "original_path": original_pth,
                "sensor_folder": best_sensor["sensor_folder"],
                "sensor_path": best_sensor["sensor_path"],
                "time_diff_sec": min_diff
            })
        else:
            results.append({
                "video_id": video_id,
                "original_path": original_pth,
                "sensor_folder": "",
                "sensor_path": "",
                "time_diff_sec": None
            })

    # 5) 输出到新的 Excel
    df_out = pd.DataFrame(results)
    out_path = "/media/robert/4TB-SSD/watchped_dataset/matched_with_merged_result.xlsx"
    df_out.to_excel(out_path, index=False)
    print(f"Done! Merged matching result saved to {out_path}")

if __name__ == "__main__":
    main()

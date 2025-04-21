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
    # 1) 读取 merged_result.xlsx (包含: video_id, original_path, date_order, ...)
    merged_excel = "/media/robert/4TB-SSD/watchped_dataset/merged_result.xlsx"
    df_merged = pd.read_excel(merged_excel)
    print("df_merged columns:", df_merged.columns.tolist())

    # 2) 读取 video_address_new.xlsx (包含: new_folder, original_path, image_number)
    #    我们想将其中的 image_number 与 merged_result 进行合并
    new_excel = "/media/robert/4TB-SSD/watchped_dataset/video_address_new.xlsx"
    df_new = pd.read_excel(new_excel)
    print("df_new columns:", df_new.columns.tolist())

    # 2.1) 把 new_folder 改名为 video_id，以便后面跟 merged_result 对上
    df_new.rename(columns={"new_folder": "video_id"}, inplace=True)

    # 2.2) 生成一个字典: video_id -> image_number
    #      这样在后续循环中，就能用 video_id 查到 image_number
    image_number_map = {}
    for idx, row in df_new.iterrows():
        vid = row["video_id"]  # 现在 rename 后, 这里就是 video_id
        img_num = row["image_number"]
        image_number_map[vid] = img_num

    # 3) 读取 Sensor 目录并解析
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

    # 4) 遍历 merged_result 每一行
    results = []
    for idx, row in df_merged.iterrows():
        # 4.1) 提取相关列
        video_id     = row["video_id"]       # e.g. "video_0001"
        original_pth = row["original_path"]  # e.g. "/media/.../Images/evening 2/01_06_2023_17_07_54_scene7..."
        date_order   = row["date_order"]     # e.g. "月日" or "日月"

        # 4.2) 查找 image_number
        image_number = image_number_map.get(video_id, None)

        # 4.3) 解析时间
        folder_name = os.path.basename(original_pth.rstrip("/"))
        dt_img = parse_folder_name(folder_name, date_order)
        if dt_img is None:
            # 解析失败
            results.append({
                "video_id": video_id,
                "original_path": original_pth,
                "sensor_folder": "",
                "sensor_path": "",
                "time_diff_sec": None,
                "time_delta_sec": None,
                "image_number": image_number
            })
            continue

        # 4.4) 匹配时间最近的 sensor
        best_sensor = None
        min_abs_diff = float('inf')
        best_diff_signed = None

        for s in sensor_info:
            diff_signed = (dt_img - s["datetime"]).total_seconds()
            abs_diff = abs(diff_signed)
            if abs_diff < min_abs_diff:
                min_abs_diff = abs_diff
                best_sensor = s
                best_diff_signed = diff_signed

        # 4.5) 存储结果
        if best_sensor:
            results.append({
                "video_id": video_id,
                "original_path": original_pth,
                "sensor_folder": best_sensor["sensor_folder"],
                "sensor_path": best_sensor["sensor_path"],
                "time_diff_sec": min_abs_diff,
                "time_delta_sec": best_diff_signed,
                "image_number": image_number
            })
        else:
            results.append({
                "video_id": video_id,
                "original_path": original_pth,
                "sensor_folder": "",
                "sensor_path": "",
                "time_diff_sec": None,
                "time_delta_sec": None,
                "image_number": image_number
            })

    # 5) 输出到新的 Excel
    df_out = pd.DataFrame(results)
    out_path = "/media/robert/4TB-SSD/watchped_dataset/matched_with_merged_result.xlsx"
    df_out.to_excel(out_path, index=False)
    print(f"Done! Merged matching result saved to {out_path}")

if __name__ == "__main__":
    main()

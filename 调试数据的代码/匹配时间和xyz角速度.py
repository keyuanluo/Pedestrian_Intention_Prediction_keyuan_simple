import os
import bisect
import pandas as pd
import datetime

def parse_gyro_formatted_time(timestr):
    """
    将 gyro 文件中的 formatted_time（例如 "27_01_2022_19_35_28_000000"）
    解析为 Python 的 datetime 对象。若解析失败返回 None。
    格式：日_月_年_时_分_秒_微秒
    """
    parts = timestr.split('_')
    if len(parts) != 7:
        return None
    try:
        day   = int(parts[0])
        month = int(parts[1])
        year  = int(parts[2])
        hour  = int(parts[3])
        minute= int(parts[4])
        second= int(parts[5])
        micro = int(parts[6])
        dt = datetime.datetime(year, month, day, hour, minute, second, micro)
        return dt
    except:
        return None

def parse_image_formatted_time(timestr):
    """
    视频/图像文件中的 image_formatted_time（例如 "06_01_2023_17_07_44_229000"）
    同样是 日_月_年_时_分_秒_微秒 的格式，也可复用同样的解析逻辑。
    """
    return parse_gyro_formatted_time(timestr)

def load_gyro_data(gyro_excel_path):
    """
    读取 gyro 文件（例如 merged_all_gyro02_sorted.xlsx），提取:
      - formatted_time
      - x, y, z
    并把 formatted_time 解析成 datetime 后按时间排序，返回列表：
    [ (dt, x, y, z, formatted_time_str), ... ]
    """
    df = pd.read_excel(gyro_excel_path)

    # 假设 gyro 文件里至少包含 ["formatted_time", "x", "y", "z"] 这几个列
    for col in ["formatted_time", "x", "y", "z"]:
        if col not in df.columns:
            raise ValueError(f"gyro 文件缺少列 '{col}'")

    gyro_list = []
    for _, row in df.iterrows():
        ft_str = str(row["formatted_time"])
        dt = parse_gyro_formatted_time(ft_str)
        if dt is None:
            continue
        x_val = row["x"]
        y_val = row["y"]
        z_val = row["z"]
        gyro_list.append((dt, x_val, y_val, z_val, ft_str))

    gyro_list.sort(key=lambda r: r[0])
    return gyro_list

def find_nearest_gyro_time(gyro_list, dt_query):
    """
    给定已按时间排序的 gyro 列表 gyro_list：[(dt, x, y, z, ft_str), ...]
    以及一个查询时间 dt_query，使用二分搜索找到时间最接近的条目，
    返回 (dt_gyro, x, y, z, formatted_time, diff_seconds)。
    若 gyro_list 为空则返回 None。
    """
    if not gyro_list:
        return None

    gyro_dts = [item[0] for item in gyro_list]
    idx = bisect.bisect_left(gyro_dts, dt_query)

    candidates = []
    if idx > 0:
        candidates.append(idx-1)
    if idx < len(gyro_list):
        candidates.append(idx)

    best = None
    best_diff = None
    for c in candidates:
        if c < 0 or c >= len(gyro_list):
            continue
        dt_gyro, x, y, z, ft_str = gyro_list[c]
        diff = abs((dt_gyro - dt_query).total_seconds())
        if (best is None) or (diff < best_diff):
            best = (dt_gyro, x, y, z, ft_str)
            best_diff = diff

    if best is None:
        return None
    dt_gyro, x, y, z, ft_str = best
    return (dt_gyro, x, y, z, ft_str, best_diff)

def merge_video_image_with_gyro(gyro_excel_path, video_image_excel_path, out_excel_path):
    """
    主函数：
    1. 读取并解析 gyro 数据
    2. 读取并解析 视频/图像 数据
    3. 对每行视频/图像数据，根据 image_formatted_time 找到最接近的 gyro 时间
    4. 合并并输出到 out_excel_path
    """
    print("[信息] 读取 GYRO 数据...")
    gyro_list = load_gyro_data(gyro_excel_path)
    print(f"[信息] gyro 数据量: {len(gyro_list)} 条")

    print("[信息] 读取 视频/图像 数据...")
    df_video = pd.read_excel(video_image_excel_path)
    needed_cols = ["video_id", "image_name", "image_original_time", "image_formatted_time"]
    for c in needed_cols:
        if c not in df_video.columns:
            raise ValueError(f"视频/图像文件缺少列 '{c}'")

    matched_gyro_time = []
    matched_x = []
    matched_y = []
    matched_z = []
    matched_diff_sec = []

    print("[信息] 开始匹配...")
    for idx, row in df_video.iterrows():
        img_ft_str = str(row["image_formatted_time"])
        dt_img = parse_image_formatted_time(img_ft_str)
        if dt_img is None:
            matched_gyro_time.append("")
            matched_x.append(float("nan"))
            matched_y.append(float("nan"))
            matched_z.append(float("nan"))
            matched_diff_sec.append(float("nan"))
            continue

        res = find_nearest_gyro_time(gyro_list, dt_img)
        if res is None:
            matched_gyro_time.append("")
            matched_x.append(float("nan"))
            matched_y.append(float("nan"))
            matched_z.append(float("nan"))
            matched_diff_sec.append(float("nan"))
        else:
            dt_gyro, x_val, y_val, z_val, ft_gyro, diff_s = res
            matched_gyro_time.append(ft_gyro)
            matched_x.append(x_val)
            matched_y.append(y_val)
            matched_z.append(z_val)
            matched_diff_sec.append(diff_s)

    df_video["gyro_formatted_time"] = matched_gyro_time
    df_video["gyro_x"] = matched_x
    df_video["gyro_y"] = matched_y
    df_video["gyro_z"] = matched_z
    df_video["time_diff_s"] = matched_diff_sec

    print(f"[信息] 完成匹配，共 {len(df_video)} 行。写出到: {out_excel_path}")
    df_video.to_excel(out_excel_path, index=False)
    print("[完成]")

if __name__ == "__main__":
    # 示例路径：将 gyro 文件路径修改为 merged_all_gyro02_sorted.xlsx
    gyro_file_path = "/media/robert/4TB-SSD/watchped_dataset/merged_all_gyro02_sorted.xlsx"
    video_file_path = "/media/robert/4TB-SSD/watchped_dataset/combined_video_image_time_fixed.xlsx"
    output_file_path = "/media/robert/4TB-SSD/watchped_dataset/combined_video_image_and_gyro_matched.xlsx"

    merge_video_image_with_gyro(
        gyro_excel_path=gyro_file_path,
        video_image_excel_path=video_file_path,
        out_excel_path=output_file_path
    )

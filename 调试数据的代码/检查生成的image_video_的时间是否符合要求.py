import os
import re
import datetime
import pandas as pd


def parse_folder_time(folder_name, date_order):
    """
    给定最后一级文件夹名(例如 "01_06_2023_17_07_54_scene7_stop_cross_veryfar") 和 date_order("月日" 或 "日月")，
    解析出 "document_original_time" (返回 datetime 或 None)。

    注意:
      - 如果是 "月日": 表示文件名格式: 月_日_年_小时_分钟_秒_...
      - 如果是 "日月": 表示文件名格式: 日_月_年_小时_分钟_秒_...

    返回一个 datetime.datetime 或 None。
    """
    # 用正则或 split 都可，这里假设前 6 段必然是 [月/日, 日/月, 年, 时, 分, 秒]，其后是场景描述等
    parts = folder_name.split("_")
    if len(parts) < 6:
        return None

    try:
        if date_order == "月日":
            # 月_日_年_时_分_秒
            month = int(parts[0])
            day = int(parts[1])
        elif date_order == "日月":
            # 日_月_年_时_分_秒
            day = int(parts[0])
            month = int(parts[1])
        else:
            print(f"[警告] 不支持的 date_order: {date_order}")
            return None

        year = int(parts[2])
        hour = int(parts[3])
        minute = int(parts[4])
        second = int(parts[5])
        return datetime.datetime(year, month, day, hour, minute, second)
    except ValueError as e:
        print(f"[警告] 解析失败: {folder_name} - {e}")
        return None


def format_datetime_dmyhms(dt):
    """
    将 datetime 转成 "日_月_年_小时_分钟_秒" (不含微秒) 例如 "06_01_2023_17_07_54"。
    如果需要带微秒，可再扩展。
    """
    return dt.strftime("%d_%m_%Y_%H_%M_%S")


def parse_image_formatted_time(image_formatted_str):
    """
    已知 image_formatted_time 类似 "06_01_2023_17_07_44_229000" 的格式:
      日_月_年_时_分_秒_微秒(6位)
    解析成 datetime.datetime 对象并返回。
    若失败返回 None。
    """
    # 先 split
    parts = image_formatted_str.split("_")
    if len(parts) < 7:
        return None

    try:
        day = int(parts[0])
        month = int(parts[1])
        year = int(parts[2])
        hour = int(parts[3])
        minute = int(parts[4])
        sec = int(parts[5])
        usec = int(parts[6])  # 微秒
        return datetime.datetime(year, month, day, hour, minute, sec, usec)
    except ValueError:
        return None


def main():
    # 1) 读取包含 [video_id, original_path, width, height, date_order] 的 Excel
    info_excel = "/media/robert/4TB-SSD/watchped_dataset/区分日月还是月日video的尺寸.xlsx"
    df_info = pd.read_excel(info_excel)

    # 2) Excel_image_时间_video_04 目录下放了许多 video_XXXX.xlsx
    #    例如: video_0001.xlsx -> 里头前几行, 其中 "000001" 行(第一行) 的 image_formatted_time
    #    我们只关心第一行 "image_formatted_time"
    excel_root = "/media/robert/4TB-SSD/watchped_dataset/Excel_image_时间_video_04"

    results = []
    for _, row in df_info.iterrows():
        video_id = row["video_id"]  # e.g. "video_0001"
        original_pth = row["original_path"]  # e.g. "/media/.../01_06_2023_17_07_54_scene7_stop_cross_veryfar"
        width = row["width"]
        height = row["height"]
        date_order = row["date_order"]  # "月日" or "日月"

        # 获取最后一级文件夹名
        folder_name = os.path.basename(original_pth.rstrip("/"))

        # (a) 解析 folder_name -> datetime
        dt_doc = parse_folder_time(folder_name, date_order)
        if dt_doc is None:
            doc_time_str = ""
        else:
            # 格式化
            doc_time_str = format_datetime_dmyhms(dt_doc)

        # (b) 读取对应 video_XXXX.xlsx, 获取第一行 "image_formatted_time"
        #     文件路径
        excel_name = f"{video_id}.xlsx"  # e.g. "video_0001.xlsx"
        excel_path = os.path.join(excel_root, excel_name)
        if not os.path.exists(excel_path):
            # 文件不存在
            print(f"[警告] {excel_path} 不存在，无法获取 image_formatted_time。")
            results.append({
                "video_id": video_id,
                "original_path": original_pth,
                "document_original_time": doc_time_str,
                "image_formatted_time": "",
                "time_diff_sec_(4-3)": ""
            })
            continue

        try:
            df_vid = pd.read_excel(excel_path)
        except Exception as e:
            print(f"[警告] 无法读取 {excel_path}: {e}")
            results.append({
                "video_id": video_id,
                "original_path": original_pth,
                "document_original_time": doc_time_str,
                "image_formatted_time": "",
                "time_diff_sec_(4-3)": ""
            })
            continue

        if df_vid.empty:
            print(f"[警告] {excel_path} 内容为空")
            results.append({
                "video_id": video_id,
                "original_path": original_pth,
                "document_original_time": doc_time_str,
                "image_formatted_time": "",
                "time_diff_sec_(4-3)": ""
            })
            continue

        # 假设前3列是: image_name, image_original_time, image_formatted_time
        # 我们只关心第一行 (index=0) 的 image_formatted_time
        row_first = df_vid.iloc[0]
        if "image_formatted_time" not in df_vid.columns:
            print(f"[警告] {excel_path} 缺少 'image_formatted_time' 列")
            img_fmt_str = ""
        else:
            img_fmt_str = str(row_first["image_formatted_time"])

        # (c) 解析 image_formatted_time
        dt_img = parse_image_formatted_time(img_fmt_str)
        # 计算差值
        if (dt_doc is not None) and (dt_img is not None):
            diff_sec = (dt_img - dt_doc).total_seconds()
        else:
            diff_sec = ""

        results.append({
            "video_id": video_id,
            "original_path": original_pth,
            "document_original_time": doc_time_str,
            "image_formatted_time": img_fmt_str,
            "time_diff_sec_(4-3)": diff_sec
        })

    # 3) 输出到新的 Excel
    df_out = pd.DataFrame(results)
    out_path = "/media/robert/4TB-SSD/watchped_dataset/time_diff_check.xlsx"
    df_out.to_excel(out_path, index=False)
    print(f"[完成] 已输出对比结果到: {out_path}")


if __name__ == "__main__":
    main()

import os
import pandas as pd
import datetime

def load_video_size_map(width_height_excel):
    """
    读取 video_width_height.xlsx，返回 { 'video_0001': (1280,720), ... } 这样的字典
    假设 excel 中有三列: ["video_id", "width", "height"].
    """
    df = pd.read_excel(width_height_excel)
    video2size = {}
    for _, row in df.iterrows():
        vid = str(row["video_id"])  # e.g. "video_0001"
        w = int(row["width"])
        h = int(row["height"])
        video2size[vid] = (w, h)
    return video2size

def parse_time_720(time_str):
    """
    解析 1280×720 情况下的时间字符串:
      例如: "1/6/23 17:07:44.229"
      => 返回 datetime 对象 (将 %y 补为 20yy)
    """
    # 用 %m/%d/%y %H:%M:%S.%f
    dt = datetime.datetime.strptime(time_str, "%m/%d/%y %H:%M:%S.%f")
    # 如果解析结果 < 2000 则 +2000
    if dt.year < 2000:
        dt = dt.replace(year=dt.year + 2000)
    return dt

def parse_time_1080(time_str):
    """
    解析 1920×1080 情况下的时间字符串:
      例如: "06/15/2022 20:33:53.090"
      => 返回 datetime 对象
    """
    # 用 %m/%d/%Y %H:%M:%S.%f (四位年份)
    dt = datetime.datetime.strptime(time_str, "%m/%d/%Y %H:%M:%S.%f")
    return dt

def format_datetime(dt):
    """
    将 datetime 转成 "日_月_年_小时_分钟_秒_微秒6位" 格式
    例如: 06_01_2023_17_44_11_000000
    """
    day = dt.day
    mon = dt.month
    year= dt.year
    hh  = dt.hour
    mm  = dt.minute
    ss  = dt.second
    us  = dt.microsecond  # 0 ~ 999999
    return f"{day:02d}_{mon:02d}_{year}_{hh:02d}_{mm:02d}_{ss:02d}_{us:06d}"

def process_excel(in_excel, out_excel, width, height):
    """
    读取 in_excel，根据 (width, height) 判断如何解析第二列时间，
    重命名列，并生成第三列 image_formatted_time，最后保存到 out_excel。

    增强：若某行时间解析失败，则若已有上一行时间，自动 +0.033333秒；否则留空。
    """
    # 读表
    df = pd.read_excel(in_excel)
    if df.empty:
        print(f"[警告] 文件 {in_excel} 内容为空，跳过。")
        return

    old_cols = df.columns.tolist()
    if len(old_cols) < 2:
        print(f"[警告] 文件 {in_excel} 列不足2列，无法处理。")
        return

    # 重命名前两列
    df.rename(columns={
        old_cols[0]: "image_name",
        old_cols[1]: "image_original_time"
    }, inplace=True)

    # 只保留前两列
    df = df[["image_name", "image_original_time"]]

    # 根据分辨率选解析函数
    if (width, height) == (1280, 720):
        parse_func = parse_time_720
    elif (width, height) == (1920, 1080):
        parse_func = parse_time_1080
    else:
        print(f"[警告] 未知分辨率 ({width}x{height})，默认当作720格式解析: {in_excel}")
        parse_func = parse_time_720

    formatted_times = []
    last_dt = None  # 用于记录上一行成功/修复后的 datetime
    time_step = datetime.timedelta(seconds=0.033333)  # ~30Hz

    for idx, row in df.iterrows():
        raw_time_str = str(row["image_original_time"]).strip()
        if not raw_time_str:
            # 空字符串 => 用 last_dt+0.033333 或留空
            if last_dt is not None:
                new_dt = last_dt + time_step
                ft = format_datetime(new_dt)
                formatted_times.append(ft)
                last_dt = new_dt
                print(f"[修正] 第{idx}行: 空字符串 => 上一行时间+0.033333 => {ft}")
            else:
                formatted_times.append("")
                print(f"[警告] 第{idx}行: 空字符串且无上一行可参考 => 留空")
            continue

        # 正常解析
        try:
            dt = parse_func(raw_time_str)
        except Exception as e:
            dt = None

        if dt is None:
            # 解析失败 => 若有 last_dt，则+0.033333，否则留空
            if last_dt is not None:
                new_dt = last_dt + time_step
                ft = format_datetime(new_dt)
                formatted_times.append(ft)
                last_dt = new_dt
                print(f"[修正] 第{idx}行: 无法解析 '{raw_time_str}', 用上一行时间+0.033333 => {ft}")
            else:
                formatted_times.append("")
                print(f"[警告] 第{idx}行: 无法解析 '{raw_time_str}', 无上一行可用 => 留空")
            continue

        # 若能正常解析
        ft = format_datetime(dt)
        formatted_times.append(ft)
        last_dt = dt

    df["image_formatted_time"] = formatted_times

    # 保存
    df.to_excel(out_excel, index=False)
    print(f"[信息] 已处理并保存: {out_excel}")

def main():
    # 1) 读取视频分辨率信息
    width_height_excel = "/media/robert/4TB-SSD/video_width_height.xlsx"
    video2size = load_video_size_map(width_height_excel)

    # 2) 输入/输出目录
    in_root = "/media/robert/4TB-SSD/watchped_dataset/Excal_image_时间_video"
    out_root= "/media/robert/4TB-SSD/watchped_dataset/Excel_image_时间_video_02"
    os.makedirs(out_root, exist_ok=True)

    # 3) 遍历
    total_processed = 0
    for fname in os.listdir(in_root):
        if not fname.lower().endswith(".xlsx"):
            continue

        base, _ = os.path.splitext(fname)  # e.g. "video_0001"
        in_excel_path = os.path.join(in_root, fname)
        out_excel_path= os.path.join(out_root, fname)

        # 查找分辨率
        if base in video2size:
            (w, h) = video2size[base]
        else:
            print(f"[警告] {base} 不在 video_width_height.xlsx 中，默认当作 (1280×720) 处理。")
            (w, h) = (1280, 720)

        process_excel(in_excel_path, out_excel_path, w, h)
        total_processed += 1

    print(f"[完成] 所有处理结束，共处理 {total_processed} 个文件。")

if __name__ == "__main__":
    import os
    main()

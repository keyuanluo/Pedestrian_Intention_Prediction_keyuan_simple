import os
import pandas as pd
import datetime
import shutil

def load_video_size_map(width_height_excel):
    """
    读取 video_width_height.xlsx，返回 { 'video_0001': (1280,720), ... } 这样的字典
    """
    df = pd.read_excel(width_height_excel)
    # 假设包含列: ["video_id", "width", "height"]
    video2size = {}
    for _, row in df.iterrows():
        vid = str(row["video_id"])     # e.g. "video_0001"
        w = int(row["width"])
        h = int(row["height"])
        video2size[vid] = (w, h)
    return video2size

def parse_time_720(time_str):
    """
    解析 1280×720 情况下的时间字符串:
      例如: "1/6/23 17:07:44.229"
      => 返回 datetime 对象 (默认年份 pivot 修正到 2000+)
    """
    # 用 %m/%d/%y %H:%M:%S.%f 解析
    # Python 对 %y 默认解析到 1900~1999，需要手动修正
    dt = datetime.datetime.strptime(time_str, "%m/%d/%y %H:%M:%S.%f")
    # 若 dt.year < 2000, 则修正 (假定都是 20xx)
    if dt.year < 2000:
        dt = dt.replace(year=dt.year + 2000)
    return dt

def parse_time_1080(time_str):
    """
    解析 1920×1080 情况下的时间字符串:
      例如: "06/15/2022 20:33:53.090"
      => 返回 datetime 对象
    """
    # 用 %m/%d/%Y %H:%M:%S.%f 解析 (四位年份)
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
    重命名列，并生成第三列 image_formatted_time，最后保存到 out_excel
    """
    # 读表
    df = pd.read_excel(in_excel)
    if df.empty:
        print(f"[警告] 文件 {in_excel} 内容为空，跳过。")
        return

    # 只关心前两列:
    #   旧表头: [ "图片名称", "第一行" ] (无论后面有没有其他列都先忽略)
    #   新表头: [ "image_name", "image_original_time" ]
    # 注: 若原文件确实只有两列，直接 rename 即可；若有多余列，后面可以 drop。
    old_cols = df.columns.tolist()
    if len(old_cols) < 2:
        print(f"[警告] 文件 {in_excel} 列不足2列，无法处理。")
        return

    # 重命名前两列
    df.rename(columns={
        old_cols[0]: "image_name",
        old_cols[1]: "image_original_time"
    }, inplace=True)

    # 若还想忽略后面所有列，可只保留这三列(后面要加新列):
    df = df[["image_name", "image_original_time"]]

    # 根据分辨率选解析函数
    if (width, height) == (1280, 720):
        parse_func = parse_time_720
    elif (width, height) == (1920, 1080):
        parse_func = parse_time_1080
    else:
        print(f"[警告] 未知分辨率 ({width}x{height})，使用默认解析(720格式)处理: {in_excel}")
        parse_func = parse_time_720

    # 生成第三列
    formatted_times = []
    for idx, row in df.iterrows():
        raw_time_str = str(row["image_original_time"])
        if not raw_time_str.strip():
            # 空字符串
            formatted_times.append("")
            continue

        try:
            dt = parse_func(raw_time_str)
            ft = format_datetime(dt)
        except Exception as e:
            print(f"  [警告] 时间解析失败: {raw_time_str} ({in_excel}) - {e}")
            ft = ""
        formatted_times.append(ft)

    df["image_formatted_time"] = formatted_times

    # 保存
    df.to_excel(out_excel, index=False)
    print(f"[信息] 已处理并保存: {out_excel}")

def main():
    # 读取分辨率信息
    width_height_excel = "/media/robert/4TB-SSD/video_width_height.xlsx"
    video2size = load_video_size_map(width_height_excel)

    # 输入目录
    in_root = "/media/robert/4TB-SSD/watchped_dataset/Excal_image_时间_video"
    # 输出目录
    out_root= "/media/robert/4TB-SSD/watchped_dataset/Excel_image_时间_video_01"
    os.makedirs(out_root, exist_ok=True)

    # 遍历
    for fname in os.listdir(in_root):
        if not fname.lower().endswith(".xlsx"):
            continue
        # 形如: video_0001.xlsx
        # 提取 video_id = "video_0001"
        base, _ = os.path.splitext(fname)  # "video_0001"
        in_excel_path = os.path.join(in_root, fname)
        out_excel_path= os.path.join(out_root, fname)

        # 查找分辨率
        if base in video2size:
            (w, h) = video2size[base]
        else:
            print(f"[警告] {base} 不在 video_width_height.xlsx 中，默认当作 1280×720 处理。")
            (w, h) = (1280, 720)

        # 处理并保存
        process_excel(in_excel_path, out_excel_path, w, h)

    print("[完成] 所有处理结束。")

if __name__ == "__main__":
    main()

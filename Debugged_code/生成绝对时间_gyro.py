import os
import pandas as pd
from datetime import datetime

def parse_folder_name(folder_name):
    """
    示例文件夹名: WEAR_27_01_2022_21_06_38
    表示: 日_月_年_时_分_秒
    返回 datetime 对象
    """
    try:
        prefix = "WEAR_"
        if not folder_name.startswith(prefix):
            print(f"[异常] 文件夹名不符合前缀 {prefix}: {folder_name}")
            return None

        parts = folder_name.replace(prefix, "").split("_")
        if len(parts) < 6:
            print(f"[异常] 文件夹名格式错误(不足6段): {folder_name}")
            return None

        day, month, year, hour, minute, sec = map(int, parts[:6])
        return datetime(year, month, day, hour, minute, sec)
    except Exception as e:
        print(f"[异常] 解析文件夹名错误: {folder_name} - {str(e)}")
        return None

def detect_delimiter(filepath):
    """
    从文件首行智能判断分隔符:
      - 若包含 ';' 则 sep=';'
      - 若包含 '\t' 则 sep='\t'
      - 否则使用空白分隔 delim_whitespace=True
    """
    if not os.path.exists(filepath):
        return None, "文件不存在"

    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            first_line = f.readline()
    except Exception as e:
        return None, f"无法读取文件首行: {e}"

    if ';' in first_line:
        return (';', None)
    elif '\t' in first_line:
        return ('\t', None)
    else:
        # 没有 ';' 也没有 '\t'，用空白分隔
        return (r'\s+', None)

def format_dmyhmsms(ts):
    """
    将 Timestamp 转成字符串: 日_月_年_小时_分钟_秒_毫秒(3位)
    例如: 27_01_2022_21_06_38_020
    """
    day    = ts.day
    month  = ts.month
    year   = ts.year
    hour   = ts.hour
    minute = ts.minute
    second = ts.second
    micro  = ts.microsecond     # 0 ~ 999999
    ms     = micro // 1000       # 毫秒 0 ~ 999
    return f"{day:02d}_{month:02d}_{year}_{hour:02d}_{minute:02d}_{second:02d}_{micro:06d}"

def process_gyro_file(folder_path, folder_time):
    """
    1) 智能检测分隔符
    2) 读取 CSV (首行为表头)
    3) 计算 absolute_time (datetime64[ns])
    4) 生成 absolute_time_str (微秒精度)
    5) 生成 formatted_time (日_月_年_小时_分钟_秒_毫秒)
    6) 输出为 Excel
    针对 WEAR_GYRO.csv 文件
    """
    gyro_file = os.path.join(folder_path, "WEAR_GYRO.csv")
    if not os.path.exists(gyro_file):
        print(f"[异常] 文件不存在: {gyro_file}")
        return

    # 1) 智能检测分隔符
    sep, err = detect_delimiter(gyro_file)
    if err:
        print(f"[异常] {err}")
        return
    if sep is None:
        print(f"[异常] 无法确定分隔符: {gyro_file}")
        return

    # 2) 读取 CSV
    try:
        if sep == r'\s+':
            df = pd.read_csv(gyro_file, delim_whitespace=True, header=0)
            print(f"[调试] 用空白分隔读取: {gyro_file}")
        else:
            df = pd.read_csv(gyro_file, sep=sep, header=0)
            print(f"[调试] 用分隔符 '{sep}' 读取: {gyro_file}")

        print("[调试] 解析到的列名:", df.columns.tolist())
        print("[调试] 前3行数据:\n", df.head(3))

    except Exception as e:
        print(f"[异常] 无法解析文件: {gyro_file} - {str(e)}")
        return

    if df.empty:
        print(f"[异常] 空文件: {gyro_file}")
        return

    if 't' not in df.columns:
        print(f"[异常] 缺少 t 列: {gyro_file}, 解析到的列名: {df.columns.tolist()}")
        return

    # 3) 获取首行 t (纳秒)
    t0_ns = df.iloc[0]['t']
    try:
        t0_ns = float(t0_ns)
    except:
        print(f"[异常] 首行 t 无法转换为数值: {gyro_file} -> {t0_ns}")
        return

    # 4) 计算每行绝对时间
    base_time = pd.Timestamp(folder_time) - pd.to_timedelta(t0_ns, unit='ns')
    try:
        df['t'] = df['t'].astype(float)
    except:
        print(f"[异常] t 列无法转换为数值: {gyro_file}")
        return

    df['absolute_time'] = base_time + pd.to_timedelta(df['t'], unit='ns')

    # 5) 生成字符串列(微秒)
    df['absolute_time_str'] = df['absolute_time'].dt.strftime("%Y-%m-%d %H:%M:%S.%f")

    # 6) 新增一列: formatted_time (日_月_年_小时_分钟_秒_毫秒)
    df['formatted_time'] = df['absolute_time'].apply(format_dmyhmsms)

    # 输出 Excel
    output_file = os.path.join(folder_path, "WEAR_GYRO_ABSOLUTE_chatgpt_01.xlsx")
    try:
        df.to_excel(output_file, index=False)
        print(f"[信息] 生成文件: {output_file}")
    except Exception as e:
        print(f"[异常] 写出 Excel 失败: {output_file} - {str(e)}")

def main():
    # 只处理单个文件夹: WEAR_27_01_2022_21_06_38
    root_dir = "/media/robert/4TB-SSD/watchped_dataset/Sensor_acc补充"
    target_folder = "WEAR_28_01_2022_11_02_10"
    folder_path = os.path.join(root_dir, target_folder)

    # 解析文件夹名 -> datetime
    folder_time = parse_folder_name(target_folder)
    if not folder_time:
        return

    # 处理 GYRO 文件
    process_gyro_file(folder_path, folder_time)

if __name__ == "__main__":
    main()

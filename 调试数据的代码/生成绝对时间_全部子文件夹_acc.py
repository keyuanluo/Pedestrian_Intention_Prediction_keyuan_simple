import os
import pandas as pd
from datetime import datetime

def parse_folder_name(folder_name):
    """
    示例: WEAR_28_01_2022_12_39_34
    -> datetime(2022, 1, 28, 12, 39, 34)
    """
    try:
        prefix = "WEAR_"
        if not folder_name.startswith(prefix):
            return None  # 不符合命名

        parts = folder_name.replace(prefix, "").split("_")
        if len(parts) < 6:
            print(f"[异常] 文件夹名格式错误: {folder_name}")
            return None

        day, month, year, hour, minute, sec = map(int, parts[:6])
        return datetime(year, month, day, hour, minute, sec)
    except Exception as e:
        print(f"[异常] 解析文件夹名错误: {folder_name} - {str(e)}")
        return None

def process_acc_file(folder_path, folder_time):
    """
    处理单个文件夹中的 WEAR_ACC.csv:
      1) 读取 CSV (分号或制表符分隔)
      2) 计算 absolute_time (datetime64[ns])
      3) 生成 absolute_time_str (微秒级字符串)
      4) 输出 WEAR_ACC_ABSOLUTE.xlsx
    """
    acc_file = os.path.join(folder_path, "WEAR_ACC.csv")
    if not os.path.exists(acc_file):
        print(f"[异常] 文件不存在: {acc_file}")
        return

    # 尝试读取
    df = None
    try:
        df = pd.read_csv(acc_file, sep=';', header=0)
    except pd.errors.ParserError:
        try:
            df = pd.read_csv(acc_file, sep='\t', header=0)
        except Exception as e:
            print(f"[异常] 无法解析文件: {acc_file} - {str(e)}")
            return

    if df is None or df.empty:
        print(f"[异常] 空文件: {acc_file}")
        return

    if 't' not in df.columns:
        print(f"[异常] 缺少 t 列: {acc_file}")
        return

    # 获取首行 t (纳秒)
    t0_ns = df.iloc[0]['t']
    try:
        t0_ns = float(t0_ns)
    except:
        print(f"[异常] 首行 t 无法转换为数值: {acc_file}")
        return

    # 将 folder_time 转成 Timestamp
    base_time = pd.Timestamp(folder_time) - pd.to_timedelta(t0_ns, unit='ns')

    # 计算每行 absolute_time
    # df['t'] 可能是字符串，需要先转 float
    try:
        df['t'] = df['t'].astype(float)
    except:
        print(f"[异常] t 列无法转换为数值: {acc_file}")
        return

    df['absolute_time'] = base_time + pd.to_timedelta(df['t'], unit='ns')

    # 生成字符串列, 微秒(6位)
    df['absolute_time_str'] = df['absolute_time'].dt.strftime("%Y-%m-%d %H:%M:%S.%f")

    # 输出 Excel
    output_file = os.path.join(folder_path, "WEAR_ACC_ABSOLUTE.xlsx")
    try:
        df.to_excel(output_file, index=False)
        print(f"[信息] 生成文件: {output_file}")
    except Exception as e:
        print(f"[异常] 写出 Excel 失败: {output_file} - {str(e)}")

def main():
    root_dir = "/media/robert/4TB-SSD/watchped_dataset/Sensor_acc补充"

    # 遍历所有子文件夹
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue  # 非文件夹跳过

        # 解析文件夹时间
        folder_time = parse_folder_name(folder_name)
        if not folder_time:
            # 若解析失败或不是 WEAR_ 前缀, 跳过
            continue

        # 处理 ACC
        process_acc_file(folder_path, folder_time)

if __name__ == "__main__":
    main()

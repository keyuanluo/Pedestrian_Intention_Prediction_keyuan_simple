import os
import pandas as pd
from datetime import datetime, timedelta


def parse_folder_name(folder_name):
    """解析文件夹时间为标准时间格式"""
    try:
        parts = folder_name.replace("WEAR_", "").split("_")
        day, month, year, hour, minute, sec = map(int, parts[:6])
        return datetime(year, month, day, hour, minute, sec)
    except:
        return None


def read_first_data_t(csv_path):
    """读取CSV文件的首行t值"""
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, sep=';', header=0, nrows=1)
        return df.iloc[0]['t'] if not df.empty else None
    except:
        return None


def main():
    root_dir = "/media/robert/4TB-SSD/watchped_dataset/Sensor (副本)"
    results = []

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # 解析文件夹时间（返回datetime对象）
        folder_dt = parse_folder_name(folder_name)
        if not folder_dt:
            continue

        # 读取传感器数据
        acc_t = read_first_data_t(os.path.join(folder_path, "WEAR_ACC.csv"))
        acg_t = read_first_data_t(os.path.join(folder_path, "WEAR_ACG.csv"))

        # 仅记录两者都有的数据
        if acc_t is not None and acg_t is not None:
            results.append({
                "folder_name": folder_name,
                "folder_time": folder_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "folder_datetime": pd.Timestamp(folder_dt),  # 转换为Timestamp
                "acc_first_t": acc_t,
                "acg_first_t": acg_t
            })

    # 转换为DataFrame
    df = pd.DataFrame(results)
    if df.empty:
        print("没有有效数据")
        return

    # 转换为秒（处理可能的类型问题）
    df['acc_first_t_s'] = pd.to_numeric(df['acc_first_t']) / 1e9
    df['acg_first_t_s'] = pd.to_numeric(df['acg_first_t']) / 1e9

    # 计算时间差（精确到微秒）
    df['time_delta'] = df['folder_datetime'] - pd.to_timedelta(df['acc_first_t_s'], unit='s')

    # 添加格式化时间差列（精确到毫秒）
    df['time_delta_formatted'] = df['time_delta'].dt.strftime("%d_%m_%Y_%H_%M_%S_") + \
                                 (df['time_delta'].dt.microsecond // 1000).astype(str).str.zfill(3)

    # 输出列排序
    output_cols = [
        'folder_name',
        'folder_time',
        'acc_first_t',
        'acg_first_t',
        'acc_first_t_s',
        'acg_first_t_s',
        'time_delta_formatted'
    ]

    # 保存结果
    output_path = "/media/robert/4TB-SSD/watchped_dataset/sensor_time_analysis.xlsx"
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        df[output_cols].to_excel(writer, index=False)

        # 设置列格式
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']

        # 秒数列格式（保留9位小数）
        number_format = workbook.add_format({'num_format': '0.000000000'})
        worksheet.set_column('E:F', 20, number_format)

        # 时间差列格式（文本格式）
        text_format = workbook.add_format({'num_format': '@'})
        worksheet.set_column('G:G', 30, text_format)

    print(f"分析结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
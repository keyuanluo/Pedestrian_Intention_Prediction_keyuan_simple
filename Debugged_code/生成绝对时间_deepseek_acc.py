# import os
# import pandas as pd
# from datetime import datetime
#
#
# def parse_folder_name(folder_name):
#     """正确解析文件夹时间（格式：WEAR_日_月_年_时_分_秒）"""
#     try:
#         parts = folder_name.replace("WEAR_", "").split("_")
#         day, month, year, hour, minute, sec = map(int, parts[:6])
#         return datetime(year, month, day, hour, minute, sec)
#     except Exception as e:
#         print(f"文件夹名解析失败: {folder_name} - {str(e)}")
#         return None
#
#
# def process_acc_file(folder_path, folder_time):
#     """处理ACC文件并生成带精确时间的Excel"""
#     acc_file = os.path.join(folder_path, "WEAR_ACC.csv")
#     output_file = os.path.join(folder_path, "WEAR_ACC_ABSOLUTE.xlsx")
#
#     if not os.path.exists(acc_file):
#         print(f"文件不存在: {acc_file}")
#         return
#
#     try:
#         # 智能检测分隔符
#         with open(acc_file, 'r') as f:
#             first_line = f.readline()
#             sep = ';' if ';' in first_line else '\t'
#
#         # 读取数据并转换类型
#         df = pd.read_csv(acc_file, sep=sep, header=0)
#         df['t'] = pd.to_numeric(df['t'], errors='coerce').astype('int64')
#
#         # 计算基准时间
#         t0_ns = df.iloc[0]['t']
#         base_time = pd.Timestamp(folder_time) - pd.to_timedelta(t0_ns, unit='ns')
#
#         # 计算绝对时间（精确到纳秒）
#         df['absolute_time'] = base_time + pd.to_timedelta(df['t'], unit='ns')
#
#         # 添加格式化字符串列（日_月_年_时_分_秒_毫秒）
#         df['formatted_time'] = df['absolute_time'].dt.strftime("%d_%m_%Y_%H_%M_%S_") + \
#                                (df['absolute_time'].dt.microsecond // 1000).astype(str).str.zfill(3)
#
#         # 保存Excel文件
#         with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
#             df.to_excel(writer, index=False)
#
#             # 获取工作簿对象
#             workbook = writer.book
#             worksheet = writer.sheets['Sheet1']
#
#             # 设置时间列格式
#             time_format = workbook.add_format({'num_format': 'dd/mm/yyyy hh:mm:ss.000000'})
#             worksheet.set_column('E:E', 30, time_format)  # 绝对时间列
#
#             # 设置格式化字符串列为文本格式
#             text_format = workbook.add_format({'num_format': '@'})
#             worksheet.set_column('F:F', 30, text_format)  # 格式化时间列
#
#             # 设置t列防止科学计数法
#             worksheet.set_column('A:A', 20, text_format)
#
#         print(f"成功生成: {output_file}")
#         print(f"首行时间示例: {df.iloc[0]['formatted_time']}")
#
#     except Exception as e:
#         print(f"处理失败: {str(e)}")
#         if 'df' in locals():
#             print("数据样例：\n", df.head(2))
#
#
# def main():
#     root_dir = "/media/robert/4TB-SSD/watchped_dataset/Sensor_acc补充"
#     target_folder = "WEAR_27_01_2022_21_06_38"
#     folder_path = os.path.join(root_dir, target_folder)
#
#     if not os.path.exists(folder_path):
#         print(f"目标文件夹不存在: {folder_path}")
#         return
#
#     folder_time = parse_folder_name(target_folder)
#     if not folder_time:
#         print("文件夹时间解析失败")
#         return
#
#     process_acc_file(folder_path, folder_time)
#
#
# if __name__ == "__main__":
#     main()

import os
import pandas as pd
from datetime import datetime


def parse_folder_name(folder_name):
    """正确解析文件夹时间（格式：WEAR_日_月_年_时_分_秒）"""
    try:
        parts = folder_name.replace("WEAR_", "").split("_")
        day, month, year, hour, minute, sec = map(int, parts[:6])
        return datetime(year, month, day, hour, minute, sec)
    except Exception as e:
        print(f"文件夹名解析失败: {folder_name} - {str(e)}")
        return None


def process_acc_file(folder_path, folder_time):
    """处理ACC文件并生成带精确时间的Excel（不包含a列）"""
    acc_file = os.path.join(folder_path, "WEAR_ACC.csv")
    output_file = os.path.join(folder_path, "WEAR_ACC_ABSOLUTE_deepseek.xlsx")

    if not os.path.exists(acc_file):
        print(f"文件不存在: {acc_file}")
        return

    try:
        # 智能检测分隔符
        with open(acc_file, 'r') as f:
            first_line = f.readline()
            sep = ';' if ';' in first_line else '\t'

        # 读取数据并转换类型
        df = pd.read_csv(acc_file, sep=sep, header=0)

        # 删除a列（如果存在）
        if 'a' in df.columns:
            df.drop(columns=['a'], inplace=True)

        df['t'] = pd.to_numeric(df['t'], errors='coerce').astype('int64')

        # 计算基准时间
        t0_ns = df.iloc[0]['t']
        base_time = pd.Timestamp(folder_time) - pd.to_timedelta(t0_ns, unit='ns')

        # 计算绝对时间（精确到纳秒）
        df['absolute_time'] = base_time + pd.to_timedelta(df['t'], unit='ns')

        # 添加格式化字符串列
        df['formatted_time'] = df['absolute_time'].dt.strftime("%d_%m_%Y_%H_%M_%S_") + \
                               (df['absolute_time'].dt.microsecond // 1000).astype(str).str.zfill(3)

        # 保存Excel文件
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)

            # 设置列格式
            workbook = writer.book
            worksheet = writer.sheets['Sheet1']

            # 时间列格式调整（删除a列后列索引改变）
            time_format = workbook.add_format({'num_format': 'dd/mm/yyyy hh:mm:ss.000000'})
            worksheet.set_column('D:D', 30, time_format)  # absolute_time现在是第5列（索引4）

            # 格式化字符串列
            text_format = workbook.add_format({'num_format': '@'})
            worksheet.set_column('E:E', 30, text_format)  # formatted_time现在是第6列（索引5）
            worksheet.set_column('A:A', 20, text_format)  # t列保持第1列

        print(f"成功生成: {output_file}")
        print("处理后的列：", df.columns.tolist())

    except Exception as e:
        print(f"处理失败: {str(e)}")
        if 'df' in locals():
            print("数据样例：\n", df.head(2))


def main():
    root_dir = "/media/robert/4TB-SSD/watchped_dataset/Sensor_acc补充"
    target_folder = "WEAR_27_01_2022_21_06_38"
    folder_path = os.path.join(root_dir, target_folder)

    if not os.path.exists(folder_path):
        print(f"目标文件夹不存在: {folder_path}")
        return

    folder_time = parse_folder_name(target_folder)
    if not folder_time:
        print("文件夹时间解析失败")
        return

    process_acc_file(folder_path, folder_time)


if __name__ == "__main__":
    main()
import os
import pandas as pd

def combine_acc_gyro_summary(acc_excel, gyro_excel, output_excel):
    """
    读取两个 Excel 文件:
      - acc_excel (如 time_diff_over_05_summary.xlsx)
      - gyro_excel (如 time_diff_over_05_summary_gyro.xlsx)
    以 video_id 为键合并，输出到 output_excel。
    """
    # 1) 读取 Excel
    df_acc = pd.read_excel(acc_excel)
    df_gyro = pd.read_excel(gyro_excel)

    # 2) 确保列名一致，如 'video_id' 和 'exceed_count'
    #    如果列名不同，需要根据实际情况调整
    #    这里假设 df_acc 有 [video_id, exceed_count]，df_gyro 同理

    # 3) 重命名列，避免合并后列冲突
    #    例如：df_acc 的 exceed_count -> exceed_count_acc
    #         df_gyro 的 exceed_count -> exceed_count_gyro
    df_acc.rename(columns={"exceed_count": "exceed_count_acc"}, inplace=True)
    df_gyro.rename(columns={"exceed_count": "exceed_count_gyro"}, inplace=True)

    # 4) 基于 'video_id' 合并
    #    how='outer' 表示取并集；若只想取交集可用 how='inner'
    df_merged = pd.merge(df_acc, df_gyro, on="video_id", how="outer")

    # 5) 输出到新的 Excel
    #    你可根据需要调整输出路径、文件名等
    os.makedirs(os.path.dirname(output_excel), exist_ok=True)
    df_merged.to_excel(output_excel, index=False)
    print(f"[信息] 已生成合并后文件: {output_excel}")


if __name__ == "__main__":
    # 示例路径（请按需修改）
    acc_excel_path = r"/media/robert/4TB-SSD/pkl运行/split_ids/time_diff_over_05_summary.xlsx"
    gyro_excel_path = r"/media/robert/4TB-SSD/pkl运行/split_ids/time_diff_over_05_summary_gyro.xlsx"
    output_path = r"/media/robert/4TB-SSD/pkl运行/split_ids/time_diff_over_05_summary_combined.xlsx"

    combine_acc_gyro_summary(acc_excel_path, gyro_excel_path, output_path)

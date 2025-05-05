import os
import pandas as pd

def check_time_diff_s_over_05_summary(excel_path):
    # 1. 读取 Excel
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"[异常] 无法读取文件: {excel_path}, 错误信息: {e}")
        return

    # 2. 检查是否存在 time_diff_s 列
    if "time_diff_s" not in df.columns:
        print("[异常] 文件中缺少 time_diff_s 列")
        return

    # 3. 筛选出 time_diff_s > 0.5 的行
    mask = df["time_diff_s"] > 0.5
    df_over_threshold = df[mask]

    if df_over_threshold.empty:
        print("[信息] 没有任何 time_diff_s > 0.5 的行")
        return

    # 4. 统计每个 video_id 中超过阈值的行数
    video_counts = df_over_threshold["video_id"].value_counts()
    # video_counts 是一个 Series，index=video_id，value=计数

    # 5. 将统计结果转换成 DataFrame
    df_summary = video_counts.reset_index()  # index -> 一列
    df_summary.columns = ["video_id", "exceed_count"]  # 重命名列

    # 6. 在终端打印出结果（可选）
    print("以下 video_id 存在 time_diff_s > 0.5 的情况：")
    for row in df_summary.itertuples(index=False):
        print(f"  - {row.video_id}: 超过阈值的行数 = {row.exceed_count}")

    # 7. 保存汇总为 Excel
    output_dir = "/media/robert/4TB-SSD/pkl运行/split_ids"
    os.makedirs(output_dir, exist_ok=True)  # 若目录不存在则创建
    output_file = os.path.join(output_dir, "time_diff_over_05_summary.xlsx")

    df_summary.to_excel(output_file, index=False)
    print(f"[信息] 已将汇总统计保存到: {output_file}")

if __name__ == "__main__":
    excel_path = r"/media/robert/4TB-SSD/watchped_dataset/combined_video_image_and_acc_matched_03.xlsx"
    check_time_diff_s_over_05_summary(excel_path)

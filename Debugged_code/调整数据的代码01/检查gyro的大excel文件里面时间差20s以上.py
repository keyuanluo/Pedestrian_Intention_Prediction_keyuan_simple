import pandas as pd

def check_time_diff_s_over_20(excel_path):
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

    # 3. 筛选出 time_diff_s > 20 的行
    mask = df["time_diff_s"] > 20
    df_over_20 = df[mask]

    if df_over_20.empty:
        print("[信息] 没有任何 time_diff_s > 20 的行")
        return

    # 4. 统计每个 video_id 中超过 20 的行数
    video_counts = df_over_20["video_id"].value_counts()

    # 5. 输出这些 video_id 及其对应行数
    print("以下 video_id 存在 time_diff_s > 20 的情况：")
    for vid, count in video_counts.items():  # 注意这里改为 .items()
        print(f"  - {vid}: 超过 20 的行数 = {count}")


if __name__ == "__main__":
    excel_path = r"/media/robert/4TB-SSD/watchped_dataset/combined_video_image_and_gyro_matched.xlsx"
    check_time_diff_s_over_20(excel_path)

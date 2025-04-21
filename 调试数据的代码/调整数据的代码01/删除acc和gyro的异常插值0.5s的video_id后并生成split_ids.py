import os
import pandas as pd
import random

def remove_and_split_videos(excel_path, output_dir):
    """
    1) 生成从 video_0001 到 video_0255 的完整列表 all_videos。
    2) 读取 excel_path（如 time_diff_over_05_summary_combined.xlsx），获取出现过的 video_id 集合 exclude_set。
    3) 从 all_videos 中移除 exclude_set，得到 remaining_videos。
    4) 随机打乱 remaining_videos，并切分为 train/val/test，比例为 70%/15%/15%。
    5) 将切分结果写到 output_dir 下的 train.txt, val.txt, test.txt。
    """

    # 1) 生成完整 video_id 列表
    all_videos = [f"video_{i:04d}" for i in range(1, 256)]  # 0001 ~ 0255

    # 2) 读取 Excel 并获取其中出现的 video_id
    df = pd.read_excel(excel_path)
    excel_videos = df["video_id"].unique().tolist()  # 出现过的video_id
    exclude_set = set(excel_videos)

    # 3) 从 all_videos 中移除 exclude_set
    remaining_videos = [v for v in all_videos if v not in exclude_set]

    # 若 remaining_videos 为空或过少，检查逻辑
    print(f"完整视频数: {len(all_videos)}")
    print(f"Excel中出现的视频数: {len(exclude_set)}")
    print(f"剩余视频数: {len(remaining_videos)}")

    if not remaining_videos:
        print("[警告] 剩余视频列表为空，无法生成 train/val/test")
        return

    # 4) 随机打乱并切分 (70% / 15% / 15%)
    random.shuffle(remaining_videos)
    total = len(remaining_videos)
    train_size = int(total * 0.70)
    val_size = int(total * 0.15)
    # test_size = total - train_size - val_size

    train_videos = remaining_videos[:train_size]
    val_videos = remaining_videos[train_size:train_size + val_size]
    test_videos = remaining_videos[train_size + val_size:]

    # 5) 输出到指定目录
    os.makedirs(output_dir, exist_ok=True)

    train_txt = os.path.join(output_dir, "train.txt")
    val_txt = os.path.join(output_dir, "val.txt")
    test_txt = os.path.join(output_dir, "test.txt")

    with open(train_txt, "w") as f:
        for vid in train_videos:
            f.write(vid + "\n")

    with open(val_txt, "w") as f:
        for vid in val_videos:
            f.write(vid + "\n")

    with open(test_txt, "w") as f:
        for vid in test_videos:
            f.write(vid + "\n")

    print(f"[信息] 已生成 train.txt/val.txt/test.txt，共 {len(remaining_videos)} 个视频。")
    print(f"  -> train: {len(train_videos)} 个")
    print(f"  -> val:   {len(val_videos)} 个")
    print(f"  -> test:  {len(test_videos)} 个")


if __name__ == "__main__":
    excel_path = r"/media/robert/4TB-SSD/pkl运行/split_ids/time_diff_over_05_summary_combined.xlsx"
    output_dir = r"/media/robert/4TB-SSD/pkl运行/split_ids/video_acc_gyro去除"

    remove_and_split_videos(excel_path, output_dir)

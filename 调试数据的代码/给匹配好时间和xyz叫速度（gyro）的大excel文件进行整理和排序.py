import os
import re
import pandas as pd

def extract_video_num(video_id):
    """
    假设 video_id 的格式形如 "video_0001"、"video_0010" 等，
    则提取后面的数字并转换为 int。
    """
    match = re.search(r"(\d+)$", video_id)
    if match:
        return int(match.group(1))
    else:
        # 若匹配不到，就返回一个较大的数，或根据需要做其他处理
        return 999999999

def main():
    # 原始文件 (GYRO 版本)
    in_excel = "/media/robert/4TB-SSD/watchped_dataset/combined_video_image_and_gyro_matched_02.xlsx"
    # 输出文件
    out_excel = "/media/robert/4TB-SSD/watchped_dataset/combined_video_image_and_gyro_matched_03.xlsx"

    # 读取原始 Excel
    df = pd.read_excel(in_excel)

    # 1) 提取 video_num
    df["video_num"] = df["video_id"].apply(extract_video_num)

    # 2) 将 image_name 转为 int，以便排序
    df["image_int"] = df["image_name"].astype(int)

    # 3) 按照 video_num (升序) + image_int (升序) 排序
    df.sort_values(by=["video_num", "image_int"], ascending=[True, True], inplace=True)

    # 4) 将 image_name 重新格式化为 6 位数字 (例如 1 -> 000001)
    df["image_name"] = df["image_int"].apply(lambda x: f"{x:06d}")

    # 5) 不再需要的中间列可删除
    df.drop(columns=["video_num", "image_int"], inplace=True)

    # 6) 保存结果
    df.to_excel(out_excel, index=False)
    print(f"[信息] 已生成新文件: {out_excel}")

if __name__ == "__main__":
    main()

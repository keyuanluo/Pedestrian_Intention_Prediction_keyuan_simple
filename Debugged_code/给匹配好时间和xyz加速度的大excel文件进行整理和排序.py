import os
import re
import pandas as pd

def extract_video_num(video_id):
    """
    假设 video_id 的格式形如 "video_0001"、"video_0010" 等，
    则提取后面的数字并转换为 int。
    """
    # 方式1: 简单分割
    # parts = video_id.split("_")  # ["video", "0001"]
    # return int(parts[1])
    #
    # 方式2: 用正则获取数字
    match = re.search(r"(\d+)$", video_id)
    if match:
        return int(match.group(1))
    else:
        # 若匹配不到，就返回一个较大的数，或根据需要做其他处理
        return 999999999

def main():
    # 假设原始文件
    in_excel = "/media/robert/4TB-SSD/watchped_dataset/combined_video_image_and_acc_matched_02.xlsx"
    # 输出文件
    out_excel = "/media/robert/4TB-SSD/watchped_dataset/combined_video_image_and_acc_matched_03.xlsx"

    # 读取原始 Excel
    df = pd.read_excel(in_excel)

    # 如果没有这两列，会报错；请确保列名一致
    # df.columns 里应该至少包含:
    #   ["video_id", "image_name", "image_original_time", "image_formatted_time", ...]
    #   以及 "acc_formatted_time", "acc_x", "acc_y", "acc_z", "time_diff_s" 等
    # 若有额外列无需删除，也可以保留

    # 1) 提取 video 的数字编号
    df["video_num"] = df["video_id"].apply(extract_video_num)

    # 2) 将 image_name 转成 int，以便后面排序
    #   注意：若原本就是字符串数字，且不会有非数字的情况，可以直接转 int
    #   若有非数字的情况，需要先清洗或做异常处理
    df["image_int"] = df["image_name"].astype(int)

    # 3) 按照 video_num (升序) + image_int (升序) 排序
    df.sort_values(by=["video_num", "image_int"], ascending=[True, True], inplace=True)

    # 4) 将 image_name 重新格式化为 6 位数字
    #    例如 1 -> 000001, 15 -> 000015
    df["image_name"] = df["image_int"].apply(lambda x: f"{x:06d}")

    # 5) 若不再需要中间列，可删除
    df.drop(columns=["video_num", "image_int"], inplace=True)

    # 6) 保存结果
    df.to_excel(out_excel, index=False)
    print(f"[信息] 已生成新文件: {out_excel}")

if __name__ == "__main__":
    main()

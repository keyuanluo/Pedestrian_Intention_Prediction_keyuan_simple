import os
import pandas as pd

def check_bbox_ranges(width_height_file, excel_folder, output_folder):
    """
    检查 excel_folder 下从 video_0001.xlsx 到 video_0255.xlsx 的文件中
    是否有 bbox 超出 video_width_height.xlsx 中对应视频的 (width, height) 范围。
    如果有，则将相关记录保存到 output_folder 下的 invalid_bbox_range.xlsx。
    """

    # 1. 读取视频分辨率映射表
    if not os.path.isfile(width_height_file):
        print(f"Error: {width_height_file} 不存在，无法读取视频宽高信息。")
        return

    df_wh = pd.read_excel(width_height_file)
    df_wh.fillna(0, inplace=True)

    # 生成 {video_0001: (1280, 720), video_0002: (1920, 1080), ...}
    video_size_map = {}
    for _, row in df_wh.iterrows():
        vid = str(row["video_id"]).strip()
        w = int(row["width"])
        h = int(row["height"])
        video_size_map[vid] = (w, h)

    # 2. 用于存放所有无效 bbox 的记录
    invalid_records = []

    # 3. 逐个文件检查
    for i in range(1, 256):
        video_str = f"video_{i:04d}"  # e.g. video_0001
        excel_path = os.path.join(excel_folder, f"{video_str}.xlsx")

        # 如果在分辨率映射表中找不到，或文件不存在，则跳过
        if video_str not in video_size_map:
            # 这里可根据需求决定是否要提示
            # print(f"{video_str} 不在分辨率表中，跳过。")
            continue
        if not os.path.isfile(excel_path):
            # 这里可根据需求决定是否要提示
            # print(f"{excel_path} 不存在，跳过。")
            continue

        w, h = video_size_map[video_str]

        # 4. 读取该视频对应的 Excel 文件
        df = pd.read_excel(excel_path)
        df.fillna(0, inplace=True)

        # 5. 找出超出范围的行
        #    条件： x1 < 0 or x2 > w or y1 < 0 or y2 > h
        mask = (
            (df["bbox_x1"] < 0) |
            (df["bbox_x2"] > w) |
            (df["bbox_y1"] < 0) |
            (df["bbox_y2"] > h)
        )
        invalid_df = df[mask].copy()

        if not invalid_df.empty:
            # 为了便于记录，补充 video_id, width, height 列
            invalid_df["video_id"] = video_str
            invalid_df["width"] = w
            invalid_df["height"] = h

            # 只保留需要的列，也可以保留全部列，根据需求来
            invalid_df = invalid_df[[
                "video_id", "image_name",
                "bbox_x1", "bbox_x2", "bbox_y1", "bbox_y2",
                "width", "height"
            ]]
            invalid_records.append(invalid_df)

    # 6. 汇总并保存
    if invalid_records:
        final_df = pd.concat(invalid_records, ignore_index=True)
        output_file = os.path.join(output_folder, "invalid_bbox_range_01.xlsx")
        final_df.to_excel(output_file, index=False)
        print(f"检测完毕，共发现 {len(final_df)} 条无效 bbox，结果已保存至: {output_file}")
    else:
        print("检测完毕，所有 bbox 均在对应分辨率范围内。未生成输出文件。")


if __name__ == "__main__":
    # 配置路径
    width_height_file = "/media/robert/4TB-SSD/video_width_height.xlsx"
    excel_folder = "/media/robert/4TB-SSD/pkl运行/Excel_融合"
    output_folder = "/media/robert/4TB-SSD"

    check_bbox_ranges(width_height_file, excel_folder, output_folder)

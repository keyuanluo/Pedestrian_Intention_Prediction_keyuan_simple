import os
import pandas as pd


def fix_bbox_ranges(width_height_file, excel_folder):
    """
    修复 excel_folder 下从 video_0001.xlsx 到 video_0255.xlsx 的文件中
    bbox_x1,bbox_x2,bbox_y1,bbox_y2 的值：
    - 若 bbox_x1 < 0，则设为 0
    - 若 bbox_y1 < 0，则设为 0
    - 若 bbox_x2 > width，则设为 width
    - 若 bbox_y2 > height，则设为 height

    修复后覆盖保存同名 Excel 文件。
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

    # 2. 遍历所有需要修复的 Excel 文件
    for i in range(1, 256):
        video_str = f"video_{i:04d}"  # e.g. video_0001
        excel_path = os.path.join(excel_folder, f"{video_str}.xlsx")

        # 如果在分辨率映射表中找不到，或文件不存在，则跳过
        if video_str not in video_size_map:
            # print(f"{video_str} 不在分辨率表中或无对应宽高，跳过。")
            continue
        if not os.path.isfile(excel_path):
            # print(f"{excel_path} 不存在，跳过。")
            continue

        w, h = video_size_map[video_str]

        # 3. 读取当前 Excel 文件
        df = pd.read_excel(excel_path)
        df.fillna(0, inplace=True)

        # 4. 修复 bbox_x1, bbox_x2, bbox_y1, bbox_y2
        #    若 < 0 则设为 0，若 > w/h 则设为 w/h
        if "bbox_x1" in df.columns and "bbox_x2" in df.columns \
                and "bbox_y1" in df.columns and "bbox_y2" in df.columns:

            # clip 函数可直接把数值裁剪到区间 [min_val, max_val]
            df["bbox_x1"] = df["bbox_x1"].clip(lower=0, upper=w)
            df["bbox_x2"] = df["bbox_x2"].clip(lower=0, upper=w)
            df["bbox_y1"] = df["bbox_y1"].clip(lower=0, upper=h)
            df["bbox_y2"] = df["bbox_y2"].clip(lower=0, upper=h)

            # 5. 覆盖写回 Excel
            df.to_excel(excel_path, index=False)
            print(f"已修复并保存 {excel_path}")
        else:
            print(f"{excel_path} 中缺少 bbox 列（bbox_x1, bbox_x2, bbox_y1, bbox_y2），跳过。")


if __name__ == "__main__":
    # 配置路径
    width_height_file = "/media/robert/4TB-SSD/video_width_height.xlsx"
    excel_folder = "/media/robert/4TB-SSD/pkl运行/Excel_融合"

    fix_bbox_ranges(width_height_file, excel_folder)

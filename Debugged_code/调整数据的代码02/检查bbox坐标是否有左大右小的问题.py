import os
import pandas as pd

def check_invalid_bbox(excel_folder, output_file="invalid_bbox_log_01.xlsx"):
    """
    检查 excel_folder 下从 video_0001.xlsx 到 video_0255.xlsx 的文件中
    是否有无效 bbox 行: (bbox_x2 - bbox_x1 < 0) 或 (bbox_y2 - bbox_y1 < 0)。
    如果存在，则将记录输出到 output_file。
    """

    invalid_records = []  # 用于收集所有无效 bbox 行

    # 遍历 video_0001.xlsx ~ video_0255.xlsx
    for i in range(1, 256):
        video_str = f"video_{i:04d}"
        excel_path = os.path.join(excel_folder, f"{video_str}.xlsx")

        if not os.path.isfile(excel_path):
            # 文件不存在则跳过
            continue

        try:
            df = pd.read_excel(excel_path)
        except Exception as e:
            print(f"读取 {excel_path} 出错: {e}")
            continue

        df.fillna(0, inplace=True)

        # 如果缺少 bbox 列名，就跳过或自行处理
        required_cols = ["bbox_x1", "bbox_x2", "bbox_y1", "bbox_y2"]
        if not all(col in df.columns for col in required_cols):
            print(f"{excel_path} 中缺少 bbox 列（bbox_x1,bbox_x2,bbox_y1,bbox_y2），跳过。")
            continue

        # 构造筛选条件： (bbox_x2 - bbox_x1 < 0) 或 (bbox_y2 - bbox_y1 < 0)
        mask = (
            (df["bbox_x2"] - df["bbox_x1"] < 0) |
            (df["bbox_y2"] - df["bbox_y1"] < 0)
        )

        invalid_df = df[mask].copy()
        if not invalid_df.empty:
            # 补充 video_id 列
            invalid_df["video_id"] = video_str
            invalid_records.append(invalid_df)

    # 汇总并保存结果
    if invalid_records:
        final_df = pd.concat(invalid_records, ignore_index=True)
        # 也可以只保留需要的列
        # columns_to_save = ["video_id", "image_name", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
        # final_df = final_df[columns_to_save]

        final_df.to_excel(output_file, index=False)
        print(f"检查完毕，共发现 {len(final_df)} 条无效 bbox 行，已记录到 {output_file}")
    else:
        print("检查完毕，未发现任何无效 bbox。")


if __name__ == "__main__":
    excel_folder = "/media/robert/4TB-SSD/pkl运行/Excel_融合"
    check_invalid_bbox(excel_folder)

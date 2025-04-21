import os
import pandas as pd


def check_speed_and_record(input_folder, output_folder):
    # 用于存放最终所有符合条件的记录
    records = []

    # 遍历从 video_0001.xlsx 到 video_0255.xlsx
    for i in range(1, 256):
        # 构造文件名，例如 video_0001.xlsx
        file_name = f"video_{i:04d}.xlsx"
        file_path = os.path.join(input_folder, file_name)

        # 如果文件存在再处理
        if os.path.isfile(file_path):
            # 从文件名里提取 video_id (例如 video_0001)
            video_id = os.path.splitext(file_name)[0]  # 去掉 .xlsx 后缀

            # 读取 Excel
            df = pd.read_excel(file_path)

            # 筛选 speed > 50 的行
            df_filtered = df[df["speed"] > 50]

            # 将符合条件的 (video_id, image_name) 加入 records
            for index, row in df_filtered.iterrows():
                image_name = row["image_name"]
                records.append({
                    "video_id": video_id,
                    "image_name": image_name
                })
        else:
            # 如果对应文件不存在，可以根据需要决定是否提示或忽略
            print(f"文件 {file_path} 不存在，跳过处理。")

    # 将所有记录转换为一个 DataFrame
    result_df = pd.DataFrame(records, columns=["video_id", "image_name"])

    # 如果有符合条件的数据，写入新的 Excel；如果没有，写一个空表或自行决定处理方式
    if not result_df.empty:
        # 将结果输出到 output_folder 下的 speed_gt_50.xlsx
        output_file = os.path.join(output_folder, "speed_gt_50.xlsx")
        result_df.to_excel(output_file, index=False)
        print(f"结果已写入 {output_file}")
    else:
        print("没有 speed > 50 的记录，未生成输出文件。")


if __name__ == "__main__":
    # 读取 Excel 文件的目录
    input_folder = "/media/robert/4TB-SSD/watchped_dataset - 副本/Excel_融合"
    # 输出结果的目录
    output_folder = "/media/robert/4TB-SSD/watchped_dataset - 副本"

    check_speed_and_record(input_folder, output_folder)

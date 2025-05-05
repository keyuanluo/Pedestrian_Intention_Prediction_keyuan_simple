import pandas as pd

def merge_excel_with_resolution(
    excel_resolution,   # Excel A: 存放 (video_id, width, height)
    excel_scenes,       # Excel B: 存放 (new_folder, original_path, ...)
    output_excel        # 输出结果的 Excel 路径
):
    """
    读取两个 Excel：
      - excel_resolution: 有列 [video_id, width, height]
      - excel_scenes: 有列 [new_folder, original_path, ...]
    最终输出到 output_excel，包含:
      [video_id, original_path, width, height, date_order]
    其中 date_order 为 "月日" 或 "日月"。
    """
    # 1) 读取两个 Excel
    df_res = pd.read_excel(excel_resolution)   # (video_id, width, height)
    df_scn = pd.read_excel(excel_scenes)       # (new_folder, original_path, ...)

    # 2) 将 df_scn 的 'new_folder' 列改名为 'video_id' 以便 merge
    df_scn.rename(columns={'new_folder': 'video_id'}, inplace=True)

    # 3) 合并，使用 'video_id' 做主键
    #    如果两个表中 video_id 不完全匹配，可以改 how='outer' 或 'left' 等
    df_merged = pd.merge(df_scn, df_res, on='video_id', how='inner')

    # 4) 根据分辨率，生成 "月日" 或 "日月"
    def get_date_order(row):
        w, h = row['width'], row['height']
        if (w == 1280 and h == 720):
            return "月日"
        elif (w == 1920 and h == 1080):
            return "日月"
        else:
            return "unknown"

    df_merged['date_order'] = df_merged.apply(get_date_order, axis=1)

    # 5) 选出并重排需要的列
    #    其中 'video_id', 'original_path', 'width', 'height', 'date_order'
    #    如果你想保留 'image_number' 等其他列，也可追加
    df_final = df_merged[['video_id', 'original_path', 'width', 'height', 'date_order']]

    # 6) 输出到 Excel
    df_final.to_excel(output_excel, index=False)
    print(f"合并完成，结果已保存到: {output_excel}")


if __name__ == "__main__":
    # 示例使用
    excel_resolution = "/media/robert/4TB-SSD/watchped_dataset/video_width_height.xlsx"   # 例如包含 [video_id, width, height]
    excel_scenes     = "/media/robert/4TB-SSD/watchped_dataset/video_address_new.xlsx"       # 例如包含 [new_folder, original_path, image_number...]
    output_excel     = "/media/robert/4TB-SSD/watchped_dataset/merged_result.xlsx"

    merge_excel_with_resolution(excel_resolution, excel_scenes, output_excel)

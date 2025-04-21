import os
import pandas as pd

def fix_image_name_format(excel_folder):
    """
    将 excel_folder 下从 video_0001.xlsx 到 video_0255.xlsx 的文件里，
    'image_name' 列改成 6 位数字（带前导零）的字符串格式，并根据
    'image_name' 升序排序后再保存。
    """

    for i in range(1, 256):
        video_str = f"video_{i:04d}"  # e.g. video_0001
        excel_path = os.path.join(excel_folder, f"{video_str}.xlsx")

        # 如果文件不存在，则跳过
        if not os.path.isfile(excel_path):
            continue

        # 读取 Excel
        df = pd.read_excel(excel_path)

        # 如果没有 image_name 列，跳过
        if "image_name" not in df.columns:
            print(f"{excel_path} 中没有 'image_name' 列，跳过。")
            continue

        # 先将 image_name 转为整数类型（若有小数，会取整）
        # 如果某些值无法转换为整数，会抛异常，需要自行处理或忽略
        try:
            df["image_name"] = df["image_name"].astype(float).astype(int)
        except ValueError as e:
            print(f"{excel_path} 中 'image_name' 列存在无法转换为整数的值，跳过。错误信息: {e}")
            continue

        # 按 image_name 升序排序
        df = df.sort_values(by="image_name", ascending=True).reset_index(drop=True)

        # 将 image_name 转为 6 位数带前导零的字符串
        df["image_name"] = df["image_name"].apply(lambda x: f"{x:06d}")

        # 覆盖保存
        df.to_excel(excel_path, index=False)
        print(f"已修正并保存 {excel_path}")

if __name__ == "__main__":
    excel_folder = "/media/robert/4TB-SSD/pkl运行/Excel_融合"
    fix_image_name_format(excel_folder)

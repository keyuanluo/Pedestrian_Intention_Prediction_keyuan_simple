import os
import shutil
import pandas as pd

def main():
    # 1) 准备路径
    excel_list_path = "/media/robert/4TB-SSD/watchped_dataset/video_address_new.xlsx"
    excel_image_time_root = "/media/robert/4TB-SSD/watchped_dataset/Excel_image_时间"
    output_root = "/media/robert/4TB-SSD/watchped_dataset/Excal_image_时间_video"

    # 若输出目录不存在，可先创建
    os.makedirs(output_root, exist_ok=True)

    # 2) 读取 video_address_new.xlsx
    df = pd.read_excel(excel_list_path)

    # 假设其中包含列: ["new_folder", "original_path", "image_number", ...]
    # 我们关注 "new_folder" 和 "original_path"
    for idx, row in df.iterrows():
        new_folder = str(row["new_folder"])       # 例如 "video_0001"
        original_pth = str(row["original_path"])  # 例如 "/media/robert/4TB-SSD/.../evening 2/01_06_2023_17_07_54_scene7_stop_cross_veryfar"

        # 3) 解析 original_path 的最后两个目录名
        #    去掉末尾斜杠后再 split
        parts = original_pth.rstrip("/").split("/")
        if len(parts) < 2:
            print(f"[警告] 行 {idx} 的 original_path 不符合预期: {original_pth}")
            continue

        subfolder_name = parts[-2]   # e.g. "evening 2"
        childfolder_name = parts[-1] # e.g. "01_06_2023_17_07_54_scene7_stop_cross_veryfar"

        # 4) 在 Excel_image_时间 下找到该文件
        #    形如: /media/robert/4TB-SSD/watchped_dataset/Excel_image_时间/子文件夹/子子文件夹.xlsx
        source_excel = os.path.join(
            excel_image_time_root,
            subfolder_name,
            childfolder_name + ".xlsx"
        )

        if not os.path.exists(source_excel):
            print(f"[警告] 未找到对应Excel: {source_excel}")
            continue

        # 5) 目标文件: /media/robert/4TB-SSD/watchped_dataset/Excal_image_时间_video/<video_0001>.xlsx
        target_excel_name = f"{new_folder}.xlsx"
        target_excel_path = os.path.join(output_root, target_excel_name)

        # 6) 复制并重命名
        try:
            shutil.copyfile(source_excel, target_excel_path)
            print(f"[信息] 已复制 {source_excel} -> {target_excel_path}")
        except Exception as e:
            print(f"[异常] 无法复制文件: {source_excel} 到 {target_excel_path} - {e}")

    print("[完成] 所有处理已结束。")

if __name__ == "__main__":
    main()

import os
import shutil
import pandas as pd

def main():
    # 1) 读取 Excel
    excel_file = "/media/robert/4TB-SSD/watchped_dataset/image和sensor完美匹配.xlsx"
    df = pd.read_excel(excel_file)

    # 2) 目标根目录
    dest_root = "/media/robert/4TB-SSD/watchped_dataset/Sensor_video"
    os.makedirs(dest_root, exist_ok=True)

    # 3) 遍历每行，复制文件夹并改名
    for idx, row in df.iterrows():
        video_id = row["video_id"]      # 例如 "video_0001"
        sensor_path = row["sensor_path"]# 例如 "/media/robert/4TB-SSD/watchped_dataset/Sensor/WEAR_06_01_2023_17_07_33"

        if not isinstance(sensor_path, str) or not sensor_path.strip():
            print(f"[{video_id}] sensor_path is empty or invalid, skip.")
            continue

        if not os.path.exists(sensor_path):
            print(f"[{video_id}] sensor_path not found: {sensor_path}")
            continue

        # 目标文件夹 => dest_root/video_0001
        new_folder = os.path.join(dest_root, video_id)

        if os.path.exists(new_folder):
            print(f"[{video_id}] target folder already exists: {new_folder}, skip or remove it first.")
            continue

        # 4) 执行复制
        try:
            # dirs_exist_ok=True (Python 3.8+) 遇到已存在的文件夹不报错
            # 如果你的 Python 版本 < 3.8，可手动判断后复制
            shutil.copytree(sensor_path, new_folder, dirs_exist_ok=False)
            print(f"[{video_id}] copied {sensor_path} => {new_folder}")
        except FileExistsError:
            print(f"[{video_id}] target folder already exists, skip.")
        except Exception as e:
            print(f"[{video_id}] copy error: {e}")

    print("All done.")

if __name__ == "__main__":
    main()

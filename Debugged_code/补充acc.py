import os
import pandas as pd


def process_acc_files(root_dir):
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)

        if not os.path.isdir(subdir_path) or not subdir.startswith("video_"):
            continue

        acc_path = os.path.join(subdir_path, "WEAR_ACC.csv")
        acg_path = os.path.join(subdir_path, "WEAR_ACG.csv")

        if os.path.exists(acc_path):
            continue

        if not os.path.exists(acg_path):
            print(f"⚠️ {subdir} 缺少 WEAR_ACG.csv")
            continue

        try:
            # 关键修改：使用分号作为分隔符
            df = pd.read_csv(acg_path, sep=';', engine='python', encoding='utf-8')

            # 清理列名中的特殊字符
            df.columns = df.columns.str.strip().str.lower()

            if 'z' not in df.columns:
                print(f"⛔ {subdir}/WEAR_ACG.csv 缺少z列")
                continue

            # 处理数据
            df['z'] = df['z'] - 9.80655

            # 保存为制表符分隔文件
            df.to_csv(acc_path, sep='\t', index=False)
            print(f"✅ 成功生成 {subdir}/WEAR_ACC.csv")

        except Exception as e:
            print(f"❌ 处理 {subdir} 时出错: {str(e)}")


if __name__ == "__main__":
    root_directory = "/media/robert/4TB-SSD/watchped_dataset/Sensor_video_acc补充"
    process_acc_files(root_directory)
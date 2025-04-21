import os

def main():
    sensor_video_root = "/media/robert/4TB-SSD/watchped_dataset/Sensor_video"

    # 用于记录缺失情况
    missing_acc = []
    missing_gyro = []
    missing_acg = []

    # 1) 遍历 sensor_video_root 下所有子文件夹
    for folder_name in os.listdir(sensor_video_root):
        folder_path = os.path.join(sensor_video_root, folder_name)

        # 确认这是一个目录
        if not os.path.isdir(folder_path):
            continue

        # 构造要检查的 csv 文件路径
        acc_file  = os.path.join(folder_path, "WEAR_ACC.csv")
        gyro_file = os.path.join(folder_path, "WEAR_GYRO.csv")
        acg_file  = os.path.join(folder_path, "WEAR_ACG.csv")

        # 检查是否缺失
        if not os.path.exists(acc_file):
            missing_acc.append(folder_name)
        if not os.path.exists(gyro_file):
            missing_gyro.append(folder_name)
        if not os.path.exists(acg_file):
            missing_acg.append(folder_name)

    # 2) 打印结果
    print("以下子文件夹缺少 WEAR_ACC.csv:")
    for f in missing_acc:
        print("  -", f)

    print("\n以下子文件夹缺少 WEAR_GYRO.csv:")
    for f in missing_gyro:
        print("  -", f)

    print("\n以下子文件夹缺少 WEAR_ACG.csv:")
    for f in missing_acg:
        print("  -", f)

if __name__ == "__main__":
    main()

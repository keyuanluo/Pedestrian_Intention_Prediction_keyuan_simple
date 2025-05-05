# import os
#
# def main():
#     sensor_root = "/media/robert/4TB-SSD/watchped_dataset/Sensor"
#
#     # 用于统计哪些子文件夹缺失 ACC 或 GYRO
#     missing_acc = []
#     missing_gyro = []
#
#     # 1) 遍历 sensor_root 下的所有条目
#     for folder_name in os.listdir(sensor_root):
#         folder_path = os.path.join(sensor_root, folder_name)
#
#         # 确认这是一个目录，而不是文件
#         if os.path.isdir(folder_path):
#             # 构造预期的 csv 文件路径
#             acc_file  = os.path.join(folder_path, "WEAR_ACC.csv")
#             gyro_file = os.path.join(folder_path, "WEAR_GYRO.csv")
#
#             # 2) 检查是否存在 WEAR_ACC.csv
#             if not os.path.exists(acc_file):
#                 missing_acc.append(folder_name)
#
#             # 3) 检查是否存在 WEAR_GYRO.csv
#             if not os.path.exists(gyro_file):
#                 missing_gyro.append(folder_name)
#
#     # 4) 打印结果
#     print("以下文件夹缺少 WEAR_ACC.csv:")
#     for f in missing_acc:
#         print("  -", f)
#
#     print("\n以下文件夹缺少 WEAR_GYRO.csv:")
#     for f in missing_gyro:
#         print("  -", f)
#
# if __name__ == "__main__":
#     main()

import os

def main():
    sensor_root = "/media/robert/4TB-SSD/watchped_dataset/Sensor"

    # 用于统计缺失情况
    missing_acc = []
    missing_gyro = []
    missing_acg = []

    # 新增：记录同时缺失的情况
    missing_both_acc_gyro = []
    missing_both_acc_acg = []

    # 1) 遍历 sensor_root 下所有子文件夹
    for folder_name in os.listdir(sensor_root):
        folder_path = os.path.join(sensor_root, folder_name)
        if not os.path.isdir(folder_path):
            continue  # 只处理文件夹

        # 构造要检查的 csv 文件路径
        acc_file  = os.path.join(folder_path, "WEAR_ACC.csv")
        gyro_file = os.path.join(folder_path, "WEAR_GYRO.csv")
        acg_file  = os.path.join(folder_path, "WEAR_ACG.csv")

        # 判断是否缺失
        acc_missing  = not os.path.exists(acc_file)
        gyro_missing = not os.path.exists(gyro_file)
        acg_missing_ = not os.path.exists(acg_file)

        if acc_missing:
            missing_acc.append(folder_name)
        if gyro_missing:
            missing_gyro.append(folder_name)
        if acg_missing_:
            missing_acg.append(folder_name)

        # 2) 判断是否“同时”缺失
        if acc_missing and gyro_missing:
            missing_both_acc_gyro.append(folder_name)
        if acc_missing and acg_missing_:
            missing_both_acc_acg.append(folder_name)

    # 3) 打印结果
    print("以下文件夹缺少 WEAR_ACC.csv:")
    for f in missing_acc:
        print("  -", f)

    print("\n以下文件夹缺少 WEAR_GYRO.csv:")
    for f in missing_gyro:
        print("  -", f)

    print("\n以下文件夹缺少 WEAR_ACG.csv:")
    for f in missing_acg:
        print("  -", f)

    print("\n以下文件夹同时缺少 WEAR_ACC.csv 和 WEAR_GYRO.csv:")
    for f in missing_both_acc_gyro:
        print("  -", f)

    print("\n以下文件夹同时缺少 WEAR_ACC.csv 和 WEAR_ACG.csv:")
    for f in missing_both_acc_acg:
        print("  -", f)

if __name__ == "__main__":
    main()

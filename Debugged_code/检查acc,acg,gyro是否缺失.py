import os
import pandas as pd

def main():
    sensor_root = "/media/robert/4TB-SSD/watchped_dataset/Sensor"
    output_excel = "/media/robert/4TB-SSD/watchped_dataset/sensor_missing_files.xlsx"

    results = []

    # 1) 遍历 sensor_root 下所有子文件夹
    for folder_name in os.listdir(sensor_root):
        folder_path = os.path.join(sensor_root, folder_name)
        if not os.path.isdir(folder_path):
            continue  # 只处理文件夹

        # 2) 构造三个文件路径
        acc_file  = os.path.join(folder_path, "WEAR_ACC.csv")
        gyro_file = os.path.join(folder_path, "WEAR_GYRO.csv")
        acg_file  = os.path.join(folder_path, "WEAR_ACG.csv")

        # 3) 判断文件是否存在
        acc_missing  = (not os.path.exists(acc_file))
        gyro_missing = (not os.path.exists(gyro_file))
        acg_missing  = (not os.path.exists(acg_file))

        # 4) 将结果存储
        results.append({
            "folder_name": folder_name,
            "acc_missing": acc_missing,
            "gyro_missing": gyro_missing,
            "acg_missing": acg_missing
        })

    # 5) 用 pandas.DataFrame 生成表格
    df = pd.DataFrame(results)

    # 如果你想把 True/False 转成 Yes/No，可以做如下替换：
    df["acc_missing"]  = df["acc_missing"].apply(lambda x: "Yes" if x else "No")
    df["gyro_missing"] = df["gyro_missing"].apply(lambda x: "Yes" if x else "No")
    df["acg_missing"]  = df["acg_missing"].apply(lambda x: "Yes" if x else "No")

    # 6) 输出到 Excel
    df.to_excel(output_excel, index=False)
    print(f"缺失文件检查完成，结果已保存到 {output_excel}")

if __name__ == "__main__":
    main()

import os
import pandas as pd


def process_wear_sensor_data(root_dir):
    """处理 WEAR_ 开头的传感器文件夹"""
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)

        # 仅处理 WEAR_ 开头的文件夹
        if not os.path.isdir(folder_path) or not folder_name.startswith("WEAR_"):
            continue

        acc_path = os.path.join(folder_path, "WEAR_ACC.csv")
        acg_path = os.path.join(folder_path, "WEAR_ACG.csv")

        # 如果 ACC 文件已存在则跳过
        if os.path.exists(acc_path):
            print(f"ℹ️ {folder_name} 已存在 WEAR_ACC.csv")
            continue

        # 检查必须的 ACG 文件
        if not os.path.exists(acg_path):
            print(f"⚠️ {folder_name} 缺少 WEAR_ACG.csv")
            continue

        try:
            # 读取并处理 ACG 数据
            df = pd.read_csv(
                acg_path,
                sep=';',  # 分号分隔符
                engine='python',  # 确保兼容性
                encoding='utf-8',  # 统一编码
                on_bad_lines='warn'  # 处理异常行
            )

            # 标准化列名（去除空格/特殊字符并小写）
            df.columns = df.columns.str.strip().str.lower()

            # 验证必要列存在
            required_columns = ['t', 'x', 'y', 'z']
            if not set(required_columns).issubset(df.columns):
                missing = set(required_columns) - set(df.columns)
                print(f"⛔ {folder_name}/WEAR_ACG.csv 缺少列: {missing}")
                continue

            # 重力补偿计算（z轴减去标准重力）
            df['z'] = df['z'] - 9.80665

            # 保存为新的 ACC 文件（制表符分隔）
            df.to_csv(acc_path, sep='\t', index=False)
            print(f"✅ 成功生成 {folder_name}/WEAR_ACC.csv")

        except Exception as e:
            print(f"❌ 处理 {folder_name} 时出错: {str(e)}")
            if 'df' in locals():
                print("样本数据预览:\n", df.head(2))


if __name__ == "__main__":
    # 新路径设置
    sensor_dir = "/media/robert/4TB-SSD/watchped_dataset/Sensor_acc补充"

    # 执行处理
    process_wear_sensor_data(sensor_dir)

    print("\n处理完成，请检查输出文件")
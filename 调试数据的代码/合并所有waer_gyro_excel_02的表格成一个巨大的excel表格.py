import os
import pandas as pd

def main():
    root_dir = "/media/robert/4TB-SSD/watchped_dataset/Sensor_acc补充"
    all_dfs = []

    # 1) 遍历所有子文件夹
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue  # 跳过非文件夹

        # 2) 在每个子文件夹中找 "WEAR_GYRO_ABSOLUTE_02.xlsx"
        gyro_file = os.path.join(folder_path, "WEAR_GYRO_ABSOLUTE_02.xlsx")
        if not os.path.exists(gyro_file):
            continue  # 没有则跳过

        # 3) 读取 Excel
        try:
            df = pd.read_excel(gyro_file)
        except Exception as e:
            print(f"[异常] 无法读取文件: {gyro_file}, 原因: {e}")
            continue

        if df.empty:
            print(f"[信息] 文件 {gyro_file} 为空，跳过")
            continue

        # 4) 将读到的 DataFrame 存储到列表中
        #    如果需要在最终大表中保留子文件夹信息，可在此加一列
        df["source_folder"] = folder_name  # 标记该行来自哪个子文件夹
        all_dfs.append(df)

    # 如果没有任何数据，直接退出
    if not all_dfs:
        print("[信息] 未收集到任何数据，退出。")
        return

    # 5) 合并所有 DataFrame
    big_df = pd.concat(all_dfs, ignore_index=True)

    # 6) 根据 absolute_time_str 排序
    #    尝试将 absolute_time_str 转换为日期时间
    big_df["abs_time_parsed"] = pd.to_datetime(big_df["absolute_time_str"],
                                               errors="coerce",
                                               format="%Y-%m-%d %H:%M:%S.%f")
    # 对解析失败的行给出提示
    failed_count = big_df["abs_time_parsed"].isna().sum()
    if failed_count > 0:
        print(f"[警告] 有 {failed_count} 行 absolute_time_str 无法转换为日期时间，可能会被排在最前或最后。")

    # 7) 对合并的大表进行排序
    big_df.sort_values(by="abs_time_parsed", inplace=True, ignore_index=True)

    # 8) 输出到一个新的 Excel
    out_path = "/media/robert/4TB-SSD/watchped_dataset/merged_all_gyro02_sorted.xlsx"
    big_df.to_excel(out_path, index=False)
    print(f"[信息] 已生成合并后的大表: {out_path}")

if __name__ == "__main__":
    main()

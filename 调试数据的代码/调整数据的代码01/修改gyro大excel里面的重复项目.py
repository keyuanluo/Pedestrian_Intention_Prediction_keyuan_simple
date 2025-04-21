import pandas as pd

def remove_repeated_gyro_values(excel_in, excel_out):
    """
    从 excel_in 中读取数据，逐行扫描 gyro_x, gyro_y, gyro_z：
      - 对第一行，直接保留原值；
      - 对后续行，若与“上一次非 0 行”的三项值完全相同，则设为 0，否则保留并更新“上一次非 0 行”的记录。
    处理后输出到 excel_out。
    """
    df = pd.read_excel(excel_in)

    required_cols = ["gyro_x", "gyro_y", "gyro_z"]
    for col in required_cols:
        if col not in df.columns:
            print(f"[异常] 文件缺少列: {col}")
            return

    # 用于记录“上一次非 0 行”的 (x, y, z)
    last_nonzero = None

    for i in range(len(df)):
        curr_x = df.at[i, "gyro_x"]
        curr_y = df.at[i, "gyro_y"]
        curr_z = df.at[i, "gyro_z"]

        if i == 0:
            # 第一行永远保留原值，更新 last_nonzero
            last_nonzero = (curr_x, curr_y, curr_z)
        else:
            # 若与上一次非 0 行相同，则置为 0
            if last_nonzero is not None:
                # 若是浮点数据且要考虑微小误差，可改用近似比较
                if (curr_x == last_nonzero[0]
                    and curr_y == last_nonzero[1]
                    and curr_z == last_nonzero[2]):
                    # 设置为 0
                    df.at[i, "gyro_x"] = 0
                    df.at[i, "gyro_y"] = 0
                    df.at[i, "gyro_z"] = 0
                else:
                    # 不同，则更新 last_nonzero
                    last_nonzero = (curr_x, curr_y, curr_z)
            else:
                # 如果 last_nonzero 还没初始化（理论上不会发生），直接更新
                last_nonzero = (curr_x, curr_y, curr_z)

    # 保存处理后的结果
    df.to_excel(excel_out, index=False)
    print(f"[信息] 已处理并保存到: {excel_out}")


if __name__ == "__main__":
    # 只保留第一次出现的三项值，其余相同值都改为 0
    excel_in  = r"/media/robert/4TB-SSD/watchped_dataset/combined_video_image_and_gyro_matched_01.xlsx"
    excel_out = r"/media/robert/4TB-SSD/watchped_dataset/combined_video_image_and_gyro_matched_02.xlsx"

    remove_repeated_gyro_values(excel_in, excel_out)

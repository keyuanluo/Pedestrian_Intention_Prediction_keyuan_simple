import pandas as pd
import numpy as np

from 测试watchped的xyz来计算对应的速度 import speeds


def main():
    input_file = "/media/robert/4TB-SSD/watchped_dataset/Sensor/WEAR_01_02_2022_13_21_02/WEAR_ACC.csv"

    # 2) 读取文件：用分号分隔，且文件本身没有表头
    #    我们手动指定列名: t, x, y, z, a
    df = pd.read_csv(input_file, sep=';', header=0, names=["t", "x", "y", "z", "a"])

    # # 检查是否读取成功
    # print("DataFrame columns:", df.columns)
    # print(df.head(5))  # 查看前5行

    # 3) 从 DataFrame 中获取时间戳和加速度
    #    这里假设时间戳在第0列，x,y,z在第1~3列；a在第4列（可不用）
    ts = df["t"].values.astype(np.float64)
    acc = df[["x", "y", "z"]].values.astype(np.float64)

    n = len(ts)
    v = np.zeros((n, 3), dtype=np.float64)  # 存放速度 (vx, vy, vz)

    # 4) 简单欧拉积分：v_{i+1} = v_i + a_i * dt
    #    dt = (t_{i+1} - t_i) × 1e-8 (假设这样得到正确秒数)
    for i in range(n - 1):
        dt = (ts[i + 1] - ts[i]) * 1e-9
        v[i + 1] = v[i] + acc[i] * dt

    speeds = np.linalg.norm(v, axis=1)


    # 5) 将结果整合到新的 DataFrame
    df_out = pd.DataFrame({
        "t": ts,
        "vx": v[:, 0],
        "vy": v[:, 1],
        "vz": v[:, 2],
        "speeds": speeds
    })

    # 5) 输出到新的 Excel 文件
    output_file = "/media/robert/4TB-SSD/watchped_dataset/Sensor/WEAR_01_02_2022_13_21_02/WEAR_ACC_velocity.xlsx"
    df_out.to_excel(output_file, index=False)

    print(f"处理完成！结果已保存到: {output_file}")


if __name__ == "__main__":
    main()

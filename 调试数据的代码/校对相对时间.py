import os
import pandas as pd
from datetime import datetime

def parse_folder_name(folder_name):
    """
    假设文件夹名形如: WEAR_28_01_2022_12_39_34
    表示 日_月_年_时_分_秒
    返回字符串 "YYYY-MM-DD HH:MM:SS" 或 None
    """
    try:
        parts = folder_name.replace("WEAR_", "").split("_")
        day   = int(parts[0])
        month = int(parts[1])
        year  = int(parts[2])
        hour  = int(parts[3])
        minute= int(parts[4])
        sec   = int(parts[5])
        dt = datetime(year, month, day, hour, minute, sec)
        return dt.strftime("%Y-%m-%d %H:%M:%S")  # 返回可读字符串
    except:
        return None

def read_first_data_t(csv_path):
    """
    读取 CSV 文件中**第一条数据行**的 t 值。
    假设:
      - 文件首行为表头: t;x;y;z;a
      - 后续行为数据行 (分号 ';' 分隔)
    如果文件不存在或解析失败, 返回 None
    """
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(
            csv_path,
            sep=';',       # 分号分隔
            header=0       # 第一行为表头 (t;x;y;z;a)
        )
        # 若成功解析, df.columns 应该是 ['t','x','y','z','a']
        if df.shape[0] == 0:
            return None
        return df.iloc[0]['t']  # 取第一条数据行的 't'
    except:
        return None

def main():
    root_dir = "/media/robert/4TB-SSD/watchped_dataset/Sensor (副本)"
    results = []

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # 1) 解析文件夹名为时间字符串 (folder_time)
        folder_time_str = parse_folder_name(folder_name)

        # 2) 读取 ACC/ACG 的第一行数据 t
        acc_file = os.path.join(folder_path, "WEAR_ACC.csv")
        acg_file = os.path.join(folder_path, "WEAR_ACG.csv")

        acc_first_t = read_first_data_t(acc_file)
        acg_first_t = read_first_data_t(acg_file)

        # 3) 暂存结果
        results.append({
            "folder_name": folder_name,
            "folder_time": folder_time_str,  # 字符串形式
            "acc_first_t": acc_first_t,      # 原始大整数 (纳秒)
            "acg_first_t": acg_first_t
        })

    # 转成 DataFrame
    df_out = pd.DataFrame(results)

    # A) 去除任何 acc_first_t 或 acg_first_t 为空的行
    df_out.replace(to_replace=[None, 'None', '', 'nan', 'NaN'], value=pd.NA, inplace=True)
    df_out.dropna(subset=["acc_first_t","acg_first_t"], inplace=True)

    # B) 将 acc_first_t / acg_first_t 转为数值, 并除以1e9 => 秒
    df_out["acc_first_t_s"] = pd.to_numeric(df_out["acc_first_t"]) / 1e9
    df_out["acg_first_t_s"] = pd.to_numeric(df_out["acg_first_t"]) / 1e9

    # C) 将 folder_time (字符串) 转为 datetime
    df_out["folder_dt"] = pd.to_datetime(df_out["folder_time"], format="%Y-%m-%d %H:%M:%S")

    # D) 计算 time_diff_s = (folder_time - acc_first_t_s)
    #    需要把 folder_dt 也转换成 Unix 时间戳(秒)
    df_out["folder_ts"] = df_out["folder_dt"].apply(lambda x: x.timestamp())
    df_out["time_diff_s"] = df_out["folder_ts"] - df_out["acc_first_t_s"]

    # E) 新建列 folder_time_s (还是 "YYYY-MM-DD HH:MM:SS")
    #    让它在表格中也显示为人类可读格式
    df_out["folder_time_s"] = df_out["folder_dt"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))

    # F) 整理列顺序
    df_out = df_out[[
        "folder_name",
        "folder_time_s",      # 保留可读格式
        "acc_first_t", "acg_first_t",
        "acc_first_t_s", "acg_first_t_s",
        "time_diff_s"
    ]]

    # G) 输出到 Excel
    out_path = "/media/robert/4TB-SSD/watchped_dataset/sensor_folder_t_check.xlsx"
    df_out.to_excel(out_path, index=False)
    print(f"结果已写入 {out_path}")

if __name__ == "__main__":
    main()

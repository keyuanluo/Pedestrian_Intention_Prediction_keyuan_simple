import os
import pandas as pd
import numpy as np

def load_acc_excel_8cols(excel_path):
    """
    读取 Excel，仅保留 8 列: [t, x, y, z, a, absolute_time, absolute_time_str, formatted_time]
    若缺列则填 NaN，多余列则忽略。
    返回 DataFrame（可能为空）。
    """
    wanted_cols = ["t", "x", "y", "z", "a", "absolute_time", "absolute_time_str", "formatted_time"]
    try:
        df_raw = pd.read_excel(excel_path)
    except Exception as e:
        print(f"[异常] 无法读取 Excel 文件: {excel_path} - {e}")
        return pd.DataFrame()  # 返回空 DataFrame

    # 重排并只保留这 8 列
    df = df_raw.reindex(columns=wanted_cols)
    return df

def detect_anomalies_two_sided(df, factor=10.0):
    """
    在只含 8 列(t,x,y,z,a,absolute_time,absolute_time_str,formatted_time)的 DataFrame 上，
    进行异常检测：
      1) 空白列：这 8 列里任意一个 NaN 则该行异常。
      2) 数值偏差（双侧）：
         - dt_prev[i] = |t[i] - t[i-1]| (i>0), 否则 NaN
         - dt_next[i] = |t[i+1] - t[i]| (i<len-1), 否则 NaN
         - 收集全部 dt_prev/dt_next（非NaN）计算 median(|dt|)
         - 若某行 i 同时满足 dt_prev[i] > threshold AND dt_next[i] > threshold，则判定“数值偏差太大”。
           (其中 threshold = factor * median(|dt|))
         - 若最前行或最后行只有一侧可比，则只要该侧 > threshold 即视为大偏差。
    返回一个带标记列的 DataFrame:
      - orig_index: 原始行号
      - dt_prev, dt_next: 双侧差值
      - missing_col: 是否空白列
      - large_diff: 是否差值过大
      - is_anomaly: 是否异常 (missing_col OR large_diff)
      - reason: 异常原因
    """
    # 复制并加一列 orig_index 记录原行号
    df = df.copy().reset_index(drop=False)
    df.rename(columns={"index": "orig_index"}, inplace=True)

    # 保证 t 可数值化
    df["t"] = pd.to_numeric(df["t"], errors="coerce")

    needed_cols = ["t", "x", "y", "z", "a", "absolute_time", "absolute_time_str", "formatted_time"]
    # 若有缺失的列先补 NaN
    for c in needed_cols:
        if c not in df.columns:
            df[c] = np.nan

    # (1) 检查空白列
    df["missing_col"] = df[needed_cols].isnull().any(axis=1)

    # (2) 计算 dt_prev / dt_next
    n = len(df)
    dt_prev = np.full(n, np.nan, dtype=float)
    dt_next = np.full(n, np.nan, dtype=float)

    for i in range(n):
        if i > 0:
            dt_prev[i] = abs(df.loc[i, "t"] - df.loc[i-1, "t"])
        if i < n - 1:
            dt_next[i] = abs(df.loc[i+1, "t"] - df.loc[i, "t"])

    df["dt_prev"] = dt_prev
    df["dt_next"] = dt_next

    # 收集非 NaN 的 dt 进入 all_dt
    all_dt = []
    all_dt.extend(dt_prev[~np.isnan(dt_prev)])
    all_dt.extend(dt_next[~np.isnan(dt_next)])
    if len(all_dt) > 0:
        typical_dt = np.median(all_dt)
    else:
        typical_dt = 0

    threshold = factor * typical_dt

    # large_diff 判断
    def check_large_diff(row):
        p = row["dt_prev"]
        q = row["dt_next"]
        if np.isnan(p) and np.isnan(q):
            return False  # 没有任何一侧 => 不判大偏差
        elif np.isnan(p):
            # 只有 next
            return (q > threshold)
        elif np.isnan(q):
            # 只有 prev
            return (p > threshold)
        else:
            # 两侧都有，需同时大于 threshold
            return (p > threshold and q > threshold)

    df["large_diff"] = df.apply(check_large_diff, axis=1)

    # 组合结果
    df["is_anomaly"] = df["missing_col"] | df["large_diff"]

    def build_reason(row):
        rs = []
        if row["missing_col"]:
            rs.append("存在空白列")
        if row["large_diff"]:
            rs.append("数值偏差太大")
        return ";".join(rs)

    df["reason"] = df.apply(build_reason, axis=1)

    return df

def check_acc_excel_two_sided(excel_path, factor=10.0):
    """
    读取 excel_path 的文件(只保留 8 列), 做 双侧差值 检测，返回:
      - df_result: 带标记列的 DataFrame
      - anom_count: 异常行数
      - total_count: 总行数
      - anom_info: [ (orig_index, t, reason), ... ] (已按 orig_index 升序)
    """
    df_8 = load_acc_excel_8cols(excel_path)
    if df_8.empty:
        return None, 0, 0, []

    df_result = detect_anomalies_two_sided(df_8, factor=factor)
    total_count = len(df_result)
    if total_count == 0:
        return df_result, 0, 0, []

    # 找出异常行
    mask_anom = df_result["is_anomaly"]
    df_anom = df_result.loc[mask_anom].copy().sort_values("orig_index")

    anom_count = len(df_anom)
    anom_info = []
    for _, row in df_anom.iterrows():
        anom_info.append((
            int(row["orig_index"]),
            row["t"],
            row["reason"]
        ))
    return df_result, anom_count, total_count, anom_info

def main():
    root_dir = "/media/robert/4TB-SSD/watchped_dataset/Sensor_acc补充"
    total_checked = 0

    # 遍历所有子文件夹
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # 只检查 "WEAR_GYRO_ABSOLUTE_01.xlsx"
        gyro_abs_path = os.path.join(folder_path, "WEAR_GYRO_ABSOLUTE_02.xlsx")
        if not os.path.exists(gyro_abs_path):
            continue

        total_checked += 1
        df_result, anom_count, total_count, anom_info = check_acc_excel_two_sided(gyro_abs_path, factor=10.0)
        if df_result is None:
            # 读取失败
            continue
        if total_count == 0:
            print(f"[信息] {gyro_abs_path}: 空或无数据")
            continue

        if anom_count == 0:
            print(f"[信息] {gyro_abs_path}: 无异常 (共 {total_count} 行)")
        else:
            print(f"[警告] {gyro_abs_path}: 发现 {anom_count} 行异常 (共 {total_count} 行)")
            print("  异常行详情(前5条):")
            for (orig_idx, t_val, reason) in anom_info[:5]:
                print(f"    -> 原行号:{orig_idx}, t={t_val}, reason={reason}")

    print("\n=== 检查完毕 ===")
    print(f"共检查到包含 WEAR_GYRO_ABSOLUTE_01.xlsx 的子文件夹数: {total_checked}")

if __name__ == "__main__":
    main()

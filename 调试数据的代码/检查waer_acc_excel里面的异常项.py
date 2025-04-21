# # # # # import os
# # # # # import pandas as pd
# # # # #
# # # # # def detect_outliers_by_diff(df, t_col='t', factor=10):
# # # # #     """
# # # # #     基于相邻行的 t 值之差来检测异常行。
# # # # #     :param df: 包含 t 列的数据 (DataFrame)
# # # # #     :param t_col: t 列名 (默认为 't')
# # # # #     :param factor: 阈值因子 (倍数)
# # # # #     :return: 标记了 'is_anomaly' 列的 DataFrame
# # # # #     """
# # # # #
# # # # #     # 如果数据还没按 t 排序，需要先排序
# # # # #     df = df.sort_values(by=t_col).reset_index(drop=True)
# # # # #
# # # # #     # 计算相邻差值 Δt_i = t_i - t_{i-1}
# # # # #     # 第一个行的 Δt 为 NaN
# # # # #     df['dt'] = df[t_col].diff()
# # # # #
# # # # #     # 计算 Δt 的“典型规模” (这里用绝对值中位数, 也可用均值或 std)
# # # # #     typical_dt = df['dt'].abs().median()
# # # # #
# # # # #     # 定义异常条件：若本行与上一行之差 > factor * typical_dt，则视为异常
# # # # #     # 你也可以用其他逻辑，如 dt < 0 或 dt 超过某阈值
# # # # #     df['is_anomaly'] = (df['dt'].abs() > factor * typical_dt)
# # # # #
# # # # #     # 返回带有 is_anomaly 列的 df
# # # # #     return df
# # # # #
# # # # # def check_t_diff_anomalies(excel_path, factor=10):
# # # # #     """
# # # # #     打开 Excel 文件, 基于相邻行 t 值差做异常检测.
# # # # #     返回 (异常行数, 总行数)
# # # # #     """
# # # # #     if not os.path.exists(excel_path):
# # # # #         print(f"[异常] 文件不存在: {excel_path}")
# # # # #         return (0, 0)
# # # # #
# # # # #     try:
# # # # #         df = pd.read_excel(excel_path)
# # # # #     except Exception as e:
# # # # #         print(f"[异常] 无法读取 {excel_path}: {e}")
# # # # #         return (0, 0)
# # # # #
# # # # #     if 't' not in df.columns:
# # # # #         print(f"[异常] 缺少 't' 列: {excel_path}")
# # # # #         return (0, len(df))
# # # # #
# # # # #     # 调用差值检测函数
# # # # #     df_result = detect_outliers_by_diff(df, t_col='t', factor=factor)
# # # # #
# # # # #     anomaly_count = df_result['is_anomaly'].sum()
# # # # #     total_count = len(df_result)
# # # # #     return (anomaly_count, total_count)
# # # # #
# # # # # def main():
# # # # #     root_dir = "/media/robert/4TB-SSD/watchped_dataset/Sensor_acc补充"
# # # # #
# # # # #     # 用于记录检测结果
# # # # #     results = []
# # # # #
# # # # #     # 遍历子文件夹
# # # # #     for folder_name in os.listdir(root_dir):
# # # # #         folder_path = os.path.join(root_dir, folder_name)
# # # # #         if not os.path.isdir(folder_path):
# # # # #             continue
# # # # #
# # # # #         excel_path = os.path.join(folder_path, "WEAR_ACC_ABSOLUTE_01.xlsx")
# # # # #         if os.path.exists(excel_path):
# # # # #             # 基于相邻差值做检测, factor=10 只是示例, 可按需调整
# # # # #             anom_count, total_count = check_t_diff_anomalies(excel_path, factor=10)
# # # # #             if total_count == 0:
# # # # #                 continue
# # # # #
# # # # #             if anom_count == 0:
# # # # #                 print(f"[信息] {excel_path}: 无异常 (共 {total_count} 行)")
# # # # #             else:
# # # # #                 print(f"[警告] {excel_path}: 发现 {anom_count} 行异常 (共 {total_count} 行)")
# # # # #
# # # # #             results.append((excel_path, anom_count, total_count))
# # # # #         else:
# # # # #             # 没有 WEAR_ACC_ABSOLUTE_01.xlsx 的文件夹跳过
# # # # #             pass
# # # # #
# # # # #     # 汇总
# # # # #     print("\n=== 汇总 ===")
# # # # #     for path, anom, total in results:
# # # # #         if anom == 0:
# # # # #             print(f"{path} -> 正常, {total} 行, 无异常")
# # # # #         else:
# # # # #             print(f"{path} -> 异常 {anom}/{total} 行")
# # # # #
# # # # # if __name__ == "__main__":
# # # # #     main()
# # # #
# # # # import os
# # # # import pandas as pd
# # # #
# # # # def detect_outliers_by_diff_and_missing(df, t_col='t', factor=10):
# # # #     """
# # # #     在原先的相邻行差值检测基础上，额外检查该行是否有空白/缺失值。
# # # #     返回带有 is_anomaly 列和 reason 列的 DataFrame。
# # # #     """
# # # #     # 1) 确保按照 t 排序（如果数据原本无序，必须先排）
# # # #     df = df.sort_values(by=t_col).reset_index(drop=False)  # 保留原来的index列，方便报告行号
# # # #     # 这里 sort_values 后的 "index" 列就是旧的行号
# # # #
# # # #     # 2) 计算相邻差值 dt = t_i - t_{i-1}
# # # #     df['dt'] = df[t_col].diff()
# # # #
# # # #     # 3) 计算“典型差值规模”（这里用绝对值中位数，也可改成均值等）
# # # #     typical_dt = df['dt'].abs().median()
# # # #
# # # #     # 4) 判断“dt 是否过大”
# # # #     #    如果 abs(dt) > factor * typical_dt，则标记“数值偏差太大”
# # # #     #    注意：第一行 dt=NaN，不参与判断
# # # #     df['large_diff'] = df['dt'].abs() > factor * typical_dt
# # # #
# # # #     # 5) 判断“是否有空白/缺失值”
# # # #     #    如果该行任意列是 NaN，则标记“存在空白列”
# # # #     #    你也可以只检查 'x','y','z' 等列
# # # #     df['missing_col'] = df.isnull().any(axis=1)
# # # #
# # # #     # 6) 组合异常标签
# # # #     #    如果 large_diff 或 missing_col 任意为 True，则此行为异常
# # # #     df['is_anomaly'] = df['large_diff'] | df['missing_col']
# # # #
# # # #     # 7) 生成 reason
# # # #     #    若该行是异常，则 reason 可能是 "数值偏差太大" / "存在空白列" / 或两者兼有
# # # #     def build_reason(row):
# # # #         reasons = []
# # # #         if row['large_diff']:
# # # #             reasons.append("数值偏差太大")
# # # #         if row['missing_col']:
# # # #             reasons.append("存在空白列")
# # # #         return ";".join(reasons)
# # # #
# # # #     df['reason'] = df.apply(build_reason, axis=1)
# # # #
# # # #     return df
# # # #
# # # #
# # # # def check_t_diff_anomalies(excel_path, factor=10):
# # # #     """
# # # #     打开 Excel 文件, 基于相邻行 t 值差 & 是否有空白列 做异常检测.
# # # #     返回 (anomalies_df, total_count)，
# # # #     其中 anomalies_df 包含所有异常行的信息(含行号、原因等)。
# # # #     """
# # # #     if not os.path.exists(excel_path):
# # # #         print(f"[异常] 文件不存在: {excel_path}")
# # # #         return pd.DataFrame(), 0
# # # #
# # # #     try:
# # # #         df = pd.read_excel(excel_path)
# # # #     except Exception as e:
# # # #         print(f"[异常] 无法读取 {excel_path}: {e}")
# # # #         return pd.DataFrame(), 0
# # # #
# # # #     if 't' not in df.columns:
# # # #         print(f"[异常] 缺少 't' 列: {excel_path}")
# # # #         return pd.DataFrame(), len(df)
# # # #
# # # #     # 调用综合检测函数
# # # #     df_result = detect_outliers_by_diff_and_missing(df, t_col='t', factor=factor)
# # # #
# # # #     # 筛选出异常行
# # # #     anomalies_df = df_result[df_result['is_anomaly']].copy()
# # # #
# # # #     # total_count 是整份表的行数
# # # #     total_count = len(df_result)
# # # #
# # # #     return anomalies_df, total_count
# # # #
# # # #
# # # # def main():
# # # #     root_dir = "/media/robert/4TB-SSD/watchped_dataset/Sensor_acc补充"
# # # #
# # # #     # 用于记录检测结果（文件路径 -> (异常行数, 总行数)）
# # # #     results = []
# # # #
# # # #     for folder_name in os.listdir(root_dir):
# # # #         folder_path = os.path.join(root_dir, folder_name)
# # # #         if not os.path.isdir(folder_path):
# # # #             continue
# # # #
# # # #         excel_path = os.path.join(folder_path, "WEAR_ACC_ABSOLUTE_01.xlsx")
# # # #         if os.path.exists(excel_path):
# # # #             # 检测
# # # #             anomalies_df, total_count = check_t_diff_anomalies(excel_path, factor=10)
# # # #             anomaly_count = len(anomalies_df)
# # # #
# # # #             if total_count == 0:
# # # #                 # 文件没数据 or 读取失败
# # # #                 continue
# # # #
# # # #             if anomaly_count == 0:
# # # #                 print(f"[信息] {excel_path}: 无异常 (共 {total_count} 行)")
# # # #             else:
# # # #                 print(f"[警告] {excel_path}: 发现 {anomaly_count} 行异常 (共 {total_count} 行)")
# # # #
# # # #                 # 如果想查看具体异常行，可以在此打印/或写出
# # # #                 # 例如只打印前5条
# # # #                 print("  异常行详情(前5条):")
# # # #                 for i, row in anomalies_df.head(5).iterrows():
# # # #                     old_idx = row['index']  # 这是原 Excel 行号(从0开始)
# # # #                     reason = row['reason']
# # # #                     t_val = row['t']
# # # #                     print(f"    -> 原行号:{old_idx}, t={t_val}, 原因={reason}")
# # # #
# # # #             results.append((excel_path, anomaly_count, total_count))
# # # #         else:
# # # #             pass  # 没有 WEAR_ACC_ABSOLUTE_01.xlsx 就跳过
# # # #
# # # #     print("\n=== 汇总 ===")
# # # #     for path, anom, total in results:
# # # #         if anom == 0:
# # # #             print(f"{path} -> 正常, {total} 行, 无异常")
# # # #         else:
# # # #             print(f"{path} -> 异常 {anom}/{total} 行")
# # # #
# # # #
# # # # if __name__ == "__main__":
# # # #     main()
# # # ###############################3
# # # # import os
# # # # import pandas as pd
# # # # import numpy as np
# # # #
# # # #
# # # # def load_and_keep_8cols(excel_path):
# # # #     """
# # # #     读取 Excel, 只保留 8 列:
# # # #       [t, x, y, z, a, absolute_time, absolute_time_str, formatted_time]
# # # #     若有缺失列则填 NaN；若有多余列则忽略。
# # # #     返回 DataFrame
# # # #     """
# # # #     wanted_cols = [
# # # #         "t", "x", "y", "z", "a",
# # # #         "absolute_time", "absolute_time_str", "formatted_time"
# # # #     ]
# # # #     df_raw = pd.read_excel(excel_path)
# # # #     df = df_raw.reindex(columns=wanted_cols)
# # # #     return df
# # # #
# # # #
# # # # def detect_anomalies_in_8cols(df, factor=10.0):
# # # #     """
# # # #     在只含 8 列(t,x,y,z,a,absolute_time,absolute_time_str,formatted_time)的 DataFrame 上，
# # # #     做简单异常检测:
# # # #       - 检测空白列(这 8 列任意一个是 NaN)
# # # #       - 对 t 列做差分, 判定是否偏差太大 (|dt| > factor * median(|dt|))
# # # #
# # # #     返回:
# # # #       df_result: 包含以下新列
# # # #          - orig_index: 原始行号
# # # #          - missing_col: 是否存在空白列 (bool)
# # # #          - dt: 与上一行的差分(排序后)
# # # #          - large_diff: 是否数值偏差过大 (bool)
# # # #          - is_anomaly: 是否异常 (bool)
# # # #          - reason: 字符串, 可能包含 "存在空白列" / "数值偏差太大"
# # # #     """
# # # #     needed_cols = ["t", "x", "y", "z", "a", "absolute_time", "absolute_time_str", "formatted_time"]
# # # #
# # # #     # 保存原始行号
# # # #     df["orig_index"] = df.index
# # # #
# # # #     # 若缺少 t 列，直接标记全部异常
# # # #     if "t" not in df.columns:
# # # #         df["missing_col"] = True
# # # #         df["dt"] = np.nan
# # # #         df["large_diff"] = True
# # # #         df["is_anomaly"] = True
# # # #         df["reason"] = "缺少 t 列"
# # # #         return df
# # # #
# # # #     # 确保 8 列都在, 不在的填 NaN
# # # #     for c in needed_cols:
# # # #         if c not in df.columns:
# # # #             df[c] = np.nan
# # # #
# # # #     # 1) 判定是否有空白列(这 8 列任意一个是 NaN)
# # # #     df["missing_col"] = df[needed_cols].isnull().any(axis=1)
# # # #
# # # #     # 2) 对 t 列做差分, 判断是否"数值偏差过大"
# # # #     #    先根据 t 排序
# # # #     df_sorted = df.sort_values("t", ignore_index=True).copy()
# # # #     df_sorted["dt"] = df_sorted["t"].diff()  # 第 0 行是 NaN
# # # #
# # # #     # 若至少有 2 行，则让第 0 行 dt = t(1) - t(0)，以免其为 NaN
# # # #     if len(df_sorted) > 1:
# # # #         df_sorted.loc[0, "dt"] = df_sorted.loc[1, "t"] - df_sorted.loc[0, "t"]
# # # #
# # # #     # 计算典型差值(中位数绝对值)
# # # #     if len(df_sorted) > 1:
# # # #         typical_dt = df_sorted["dt"].iloc[1:].abs().median()  # 跳过第 0 行
# # # #     else:
# # # #         typical_dt = 0
# # # #
# # # #     df_sorted["large_diff"] = False
# # # #     if typical_dt > 0:
# # # #         df_sorted["large_diff"] = df_sorted["dt"].abs() > factor * typical_dt
# # # #
# # # #     # ============ 关键改动：忽略排序后第 0 行的“数值偏差太大” ============
# # # #     if len(df_sorted) > 0:
# # # #         df_sorted.loc[0, "large_diff"] = False
# # # #         df_sorted.loc[0, "dt"] = 0
# # # #     # =================================================================
# # # #
# # # #     # 把 dt 和 large_diff 合并回原 df(通过 orig_index 对应)
# # # #     df_sorted.set_index("orig_index", inplace=True)
# # # #     df = df.set_index("orig_index")
# # # #     df["dt"] = df_sorted["dt"]
# # # #     df["large_diff"] = df_sorted["large_diff"]
# # # #     df.reset_index(inplace=True)  # 恢复
# # # #
# # # #     # 3) 生成 is_anomaly & reason
# # # #     df["is_anomaly"] = df["missing_col"] | df["large_diff"]
# # # #
# # # #     def build_reason(row):
# # # #         rs = []
# # # #         if row["missing_col"]:
# # # #             rs.append("存在空白列")
# # # #         if row["large_diff"]:
# # # #             rs.append("数值偏差太大")
# # # #         return ";".join(rs)
# # # #
# # # #     df["reason"] = df.apply(build_reason, axis=1)
# # # #
# # # #     return df
# # # #
# # # #
# # # # def check_acc_excel(excel_path, factor=10.0):
# # # #     """
# # # #     检查单个 WEAR_ACC_ABSOLUTE_01.xlsx 文件:
# # # #       1) 只保留 8 列
# # # #       2) 做异常检测
# # # #       3) 返回 (df_result, anomalies_count, total_count, anomaly_rows_info)
# # # #
# # # #     其中 anomaly_rows_info 是一个列表, 每项是 (orig_index, t值, reason)
# # # #     """
# # # #     df = load_and_keep_8cols(excel_path)
# # # #     if df.empty:
# # # #         # 返回空，表示无法读取或为空
# # # #         return None, 0, 0, []
# # # #
# # # #     df_result = detect_anomalies_in_8cols(df, factor=factor)
# # # #     total_count = len(df_result)
# # # #
# # # #     # 找出异常行
# # # #     mask_anom = df_result["is_anomaly"] == True
# # # #     anomalies_count = mask_anom.sum()
# # # #     if anomalies_count == 0:
# # # #         return df_result, 0, total_count, []
# # # #
# # # #     # 取异常行信息
# # # #     df_anom = df_result.loc[mask_anom].sort_values("orig_index")
# # # #     anomaly_rows_info = []
# # # #     for _, row in df_anom.iterrows():
# # # #         anomaly_rows_info.append((
# # # #             int(row["orig_index"]),
# # # #             row["t"],
# # # #             row["reason"]
# # # #         ))
# # # #     return df_result, anomalies_count, total_count, anomaly_rows_info
# # # #
# # # #
# # # # def main():
# # # #     root_dir = "/media/robert/4TB-SSD/watchped_dataset/Sensor_acc补充"
# # # #     # 只检查一层子目录
# # # #     for folder_name in os.listdir(root_dir):
# # # #         folder_path = os.path.join(root_dir, folder_name)
# # # #         if os.path.isdir(folder_path) and folder_name.startswith("WEAR_"):
# # # #             acc_abs_path = os.path.join(folder_path, "WEAR_ACC_ABSOLUTE_01.xlsx")
# # # #             if not os.path.exists(acc_abs_path):
# # # #                 continue  # 子文件夹没有这个文件则跳过
# # # #
# # # #             df_result, anom_count, total_count, anom_info = check_acc_excel(acc_abs_path, factor=10.0)
# # # #             if df_result is None:
# # # #                 print(f"[警告] {acc_abs_path}: 文件为空或读取失败")
# # # #                 continue
# # # #
# # # #             if anom_count == 0:
# # # #                 print(f"[信息] {acc_abs_path}: 无异常 (共 {total_count} 行)")
# # # #             else:
# # # #                 print(f"[警告] {acc_abs_path}: 发现 {anom_count} 行异常 (共 {total_count} 行)")
# # # #                 # 只打印前5条详情
# # # #                 max_show = 5
# # # #                 print("  异常行详情(前5条):")
# # # #                 for i, (orig_idx, t_val, reason) in enumerate(anom_info):
# # # #                     if i >= max_show:
# # # #                         break
# # # #                     print(f"    -> 原行号:{orig_idx}, t={t_val}, reason={reason}")
# # # #
# # # #     print("=== 检查完毕 ===")
# # # #
# # # #
# # # # if __name__ == "__main__":
# # # #     main()
# # #
# # #
# # # import os
# # # import pandas as pd
# # # import numpy as np
# # #
# # # def load_and_keep_8cols(excel_path):
# # #     """
# # #     读取 Excel, 只保留 8 列:
# # #       [t, x, y, z, a, absolute_time, absolute_time_str, formatted_time]
# # #     若有缺失列则填 NaN；若有多余列则忽略。
# # #     返回 DataFrame
# # #     """
# # #     wanted_cols = [
# # #         "t", "x", "y", "z", "a",
# # #         "absolute_time", "absolute_time_str", "formatted_time"
# # #     ]
# # #     df_raw = pd.read_excel(excel_path)
# # #     df = df_raw.reindex(columns=wanted_cols)
# # #     return df
# # #
# # #
# # # def detect_anomalies_in_8cols(df, factor=10.0):
# # #     """
# # #     在只含 8 列(t,x,y,z,a,absolute_time,absolute_time_str,formatted_time)的 DataFrame 上，
# # #     做简单异常检测:
# # #       - 检测空白列(这 8 列任意一个是 NaN)
# # #       - 对 t 列做差分, 判定是否偏差太大 (|dt| > factor * median(|dt|)),
# # #         但会自动忽略 "最小 t 行" 的数值偏差。
# # #
# # #     返回包含以下新列:
# # #        - orig_index: 原始行号(方便追溯)
# # #        - missing_col: 是否存在空白列 (bool)
# # #        - dt: 与上一行的差分(按 t 排序后)
# # #        - large_diff: 是否数值偏差太大 (bool)
# # #        - is_anomaly: 是否异常 (bool)
# # #        - reason: 字符串, 可能包含 "存在空白列" / "数值偏差太大"
# # #     """
# # #     needed_cols = ["t","x","y","z","a","absolute_time","absolute_time_str","formatted_time"]
# # #
# # #     # 记录原始行号 (方便后面合并)
# # #     df["orig_index"] = df.index
# # #
# # #     # 若缺少 t 列，直接标记全部异常
# # #     if "t" not in df.columns:
# # #         df["missing_col"] = True
# # #         df["dt"] = np.nan
# # #         df["large_diff"] = True
# # #         df["is_anomaly"] = True
# # #         df["reason"] = "缺少 t 列"
# # #         return df
# # #
# # #     # 确保 needed_cols 都在
# # #     for c in needed_cols:
# # #         if c not in df.columns:
# # #             df[c] = np.nan
# # #
# # #     # 1) 判定是否有空白列: 这 8 列里只要有 NaN 就视为 missing_col=True
# # #     df["missing_col"] = df[needed_cols].isnull().any(axis=1)
# # #
# # #     # 2) 对 t 列做差分, 判断是否"数值偏差过大"
# # #     #    (1) 根据 t 排序，但保留原索引
# # #     df_sorted = df.sort_values("t", ascending=True, inplace=False)  # 不忽略原索引
# # #
# # #     #    (2) 计算 dt
# # #     df_sorted["dt"] = df_sorted["t"].diff()  # 最小 t 的行 dt=NaN
# # #
# # #     #    (3) 若至少有 2 行, 则给第一行(最小 t 行)一个 dt=0
# # #     #         (这样它不会出现 "NaN" 差分)
# # #     if len(df_sorted) > 1:
# # #         first_idx = df_sorted.index[0]  # 最小 t 所在行的 原始 index
# # #         second_idx = df_sorted.index[1]
# # #         # dt = t(1) - t(0) 也可以，但常见做法是直接设 0
# # #         df_sorted.loc[first_idx, "dt"] = 0
# # #
# # #     #    (4) 计算典型差值(中位数绝对值)
# # #     if len(df_sorted) > 2:
# # #         # 跳过第一行后(因为第一行 dt=0), 用后面行的 dt 做中位数
# # #         typical_dt = df_sorted["dt"].iloc[1:].abs().median()
# # #     else:
# # #         typical_dt = 0
# # #
# # #     #    (5) large_diff：|dt| > factor * typical_dt
# # #     df_sorted["large_diff"] = False
# # #     if typical_dt > 0:
# # #         mask = df_sorted["dt"].abs() > factor * typical_dt
# # #         df_sorted.loc[mask, "large_diff"] = True
# # #
# # #     #    (6) 同时，为了不把最小 t 行判为 large_diff=True，我们再强制最小 t 行 large_diff=False
# # #     if len(df_sorted) > 0:
# # #         first_idx = df_sorted.index[0]
# # #         df_sorted.loc[first_idx, "large_diff"] = False
# # #
# # #     # 3) 合并回原 df
# # #     #    (1) 先把 dt, large_diff 合并回去
# # #     #         df_sorted 里 dt/large_diff 对应原索引
# # #     df = df.set_index("orig_index")  # 以原索引为 index
# # #     df_sorted = df_sorted.set_index("orig_index")  # 同样
# # #     df["dt"] = df_sorted["dt"]
# # #     df["large_diff"] = df_sorted["large_diff"]
# # #
# # #     #    (2) 恢复 numeric index
# # #     df.reset_index(inplace=True)
# # #
# # #     # 4) 生成 is_anomaly & reason
# # #     df["is_anomaly"] = df["missing_col"] | df["large_diff"]
# # #
# # #     def build_reason(row):
# # #         rs = []
# # #         if row["missing_col"]:
# # #             rs.append("存在空白列")
# # #         if row["large_diff"]:
# # #             rs.append("数值偏差太大")
# # #         return ";".join(rs)
# # #
# # #     df["reason"] = df.apply(build_reason, axis=1)
# # #     return df
# # #
# # #
# # # def check_acc_excel(excel_path, factor=10.0):
# # #     """
# # #     检查单个 WEAR_ACC_ABSOLUTE_01.xlsx 文件:
# # #       1) 只保留 8 列
# # #       2) 做异常检测
# # #       3) 返回 (df_result, anomalies_count, total_count, anomaly_rows_info)
# # #
# # #     其中 anomaly_rows_info 是一个列表, 每项是 (orig_index, t值, reason)
# # #     """
# # #     df = load_and_keep_8cols(excel_path)
# # #     if df.empty:
# # #         # 返回空，表示无法读取或为空
# # #         return None, 0, 0, []
# # #
# # #     df_result = detect_anomalies_in_8cols(df, factor=factor)
# # #     total_count = len(df_result)
# # #
# # #     # 找出异常行
# # #     mask_anom = df_result["is_anomaly"] == True
# # #     anomalies_count = mask_anom.sum()
# # #     if anomalies_count == 0:
# # #         return df_result, 0, total_count, []
# # #
# # #     # 取异常行信息
# # #     df_anom = df_result.loc[mask_anom].sort_values("index")  # or sort_values("orig_index")
# # #     anomaly_rows_info = []
# # #     for _, row in df_anom.iterrows():
# # #         anomaly_rows_info.append((
# # #             int(row["index"]),  # row["index"]就是pandas默认行号; 若要用orig_index则看情况
# # #             row["t"],
# # #             row["reason"]
# # #         ))
# # #     return df_result, anomalies_count, total_count, anomaly_rows_info
# # #
# # #
# # # def main():
# # #     root_dir = "/media/robert/4TB-SSD/watchped_dataset/Sensor_acc补充"
# # #
# # #     # 遍历每个子文件夹
# # #     for folder_name in os.listdir(root_dir):
# # #         folder_path = os.path.join(root_dir, folder_name)
# # #         # 只处理形如 WEAR_ 开头的文件夹
# # #         if not (os.path.isdir(folder_path) and folder_name.startswith("WEAR_")):
# # #             continue
# # #
# # #         acc_abs_path = os.path.join(folder_path, "WEAR_ACC_ABSOLUTE_01.xlsx")
# # #         if not os.path.exists(acc_abs_path):
# # #             # 子文件夹没有这个文件则跳过
# # #             continue
# # #
# # #         df_result, anom_count, total_count, anom_info = check_acc_excel(acc_abs_path, factor=10.0)
# # #         if df_result is None:
# # #             print(f"[警告] {acc_abs_path}: 文件为空或读取失败")
# # #             continue
# # #
# # #         if anom_count == 0:
# # #             print(f"[信息] {acc_abs_path}: 无异常 (共 {total_count} 行)")
# # #         else:
# # #             print(f"[警告] {acc_abs_path}: 发现 {anom_count} 行异常 (共 {total_count} 行)")
# # #             max_show = 5
# # #             print("  异常行详情(前5条):")
# # #             for i, (row_idx, t_val, reason) in enumerate(anom_info):
# # #                 if i >= max_show:
# # #                     break
# # #                 print(f"    -> 原行号:{row_idx}, t={t_val}, reason={reason}")
# # #
# # #     print("=== 检查完毕 ===")
# # #
# # #
# # # if __name__ == "__main__":
# # #     main()
# #
# # import os
# # import pandas as pd
# # import numpy as np
# #
# # def load_and_keep_8cols(excel_path):
# #     """
# #     读取 Excel, 只保留 8 列:
# #       [t, x, y, z, a, absolute_time, absolute_time_str, formatted_time]
# #     若有缺失列则填 NaN；若有多余列则忽略。
# #     返回 DataFrame
# #     """
# #     wanted_cols = [
# #         "t", "x", "y", "z", "a",
# #         "absolute_time", "absolute_time_str", "formatted_time"
# #     ]
# #     df_raw = pd.read_excel(excel_path)
# #     df = df_raw.reindex(columns=wanted_cols)
# #     return df
# #
# #
# # def detect_anomalies_in_8cols(df, factor=10.0):
# #     """
# #     在只含 8 列(t,x,y,z,a,absolute_time,absolute_time_str,formatted_time)的 DataFrame 上，
# #     做简单异常检测:
# #       - 检测空白列(这 8 列任意一个是 NaN)
# #       - 对 t 列做差分, 判定是否偏差太大 (|dt| > factor * median(|dt|)),
# #         但会自动忽略 "最小 t 行" 的数值偏差。
# #
# #     返回包含以下新列:
# #        - orig_index: 原始行号(方便追溯)
# #        - missing_col: 是否存在空白列 (bool)
# #        - dt: 与上一行的差分(按 t 排序后)
# #        - large_diff: 是否数值偏差太大 (bool)
# #        - is_anomaly: 是否异常 (bool)
# #        - reason: 字符串, 可能包含 "存在空白列" / "数值偏差太大"
# #     """
# #     needed_cols = ["t","x","y","z","a","absolute_time","absolute_time_str","formatted_time"]
# #
# #     # 记录原始行号 (方便后面合并)
# #     df["orig_index"] = df.index
# #
# #     # 若缺少 t 列，直接标记全部异常
# #     if "t" not in df.columns:
# #         df["missing_col"] = True
# #         df["dt"] = np.nan
# #         df["large_diff"] = True
# #         df["is_anomaly"] = True
# #         df["reason"] = "缺少 t 列"
# #         return df
# #
# #     # 确保 needed_cols 都在
# #     for c in needed_cols:
# #         if c not in df.columns:
# #             df[c] = np.nan
# #
# #     # 1) 判定是否有空白列: 这 8 列里只要有 NaN 就视为 missing_col=True
# #     df["missing_col"] = df[needed_cols].isnull().any(axis=1)
# #
# #     # 2) 对 t 列做差分, 判断是否"数值偏差过大"
# #     #    (1) 根据 t 排序，但保留原索引
# #     df_sorted = df.sort_values("t", ascending=True, inplace=False)  # 不忽略原索引
# #
# #     #    (2) 计算 dt
# #     df_sorted["dt"] = df_sorted["t"].diff()  # 最小 t 的行 dt=NaN
# #
# #     #    (3) 若至少有 2 行, 则给第一行(最小 t 行)一个 dt=0
# #     #         (这样它不会出现 "NaN" 差分)
# #     if len(df_sorted) > 1:
# #         first_idx = df_sorted.index[0]  # 最小 t 所在行的 原始 index
# #         second_idx = df_sorted.index[1]
# #         # dt = t(1) - t(0) 也可以，但常见做法是直接设 0
# #         df_sorted.loc[first_idx, "dt"] = 0
# #
# #     #    (4) 计算典型差值(中位数绝对值)
# #     if len(df_sorted) > 2:
# #         # 跳过第一行后(因为第一行 dt=0), 用后面行的 dt 做中位数
# #         typical_dt = df_sorted["dt"].iloc[1:].abs().median()
# #     else:
# #         typical_dt = 0
# #
# #     #    (5) large_diff：|dt| > factor * typical_dt
# #     df_sorted["large_diff"] = False
# #     if typical_dt > 0:
# #         mask = df_sorted["dt"].abs() > factor * typical_dt
# #         df_sorted.loc[mask, "large_diff"] = True
# #
# #     #    (6) 同时，为了不把最小 t 行判为 large_diff=True，我们再强制最小 t 行 large_diff=False
# #     if len(df_sorted) > 0:
# #         first_idx = df_sorted.index[0]
# #         df_sorted.loc[first_idx, "large_diff"] = False
# #
# #     # 3) 合并回原 df
# #     #    (1) 先把 dt, large_diff 合并回去
# #     #         df_sorted 里 dt/large_diff 对应原索引
# #     df = df.set_index("orig_index")  # 以原索引为 index
# #     df_sorted = df_sorted.set_index("orig_index")  # 同样
# #     df["dt"] = df_sorted["dt"]
# #     df["large_diff"] = df_sorted["large_diff"]
# #
# #     #    (2) 恢复 numeric index
# #     df.reset_index(inplace=True)
# #
# #     # 4) 生成 is_anomaly & reason
# #     df["is_anomaly"] = df["missing_col"] | df["large_diff"]
# #
# #     def build_reason(row):
# #         rs = []
# #         if row["missing_col"]:
# #             rs.append("存在空白列")
# #         if row["large_diff"]:
# #             rs.append("数值偏差太大")
# #         return ";".join(rs)
# #
# #     df["reason"] = df.apply(build_reason, axis=1)
# #     return df
# #
# #
# # def check_acc_excel(excel_path, factor=10.0):
# #     """
# #     检查单个 WEAR_ACC_ABSOLUTE_01.xlsx 文件:
# #       1) 只保留 8 列
# #       2) 做异常检测
# #       3) 返回 (df_result, anomalies_count, total_count, anomaly_rows_info)
# #
# #     其中 anomaly_rows_info 是一个列表, 每项是 (orig_index, t值, reason)
# #     """
# #     df = load_and_keep_8cols(excel_path)
# #     if df.empty:
# #         # 返回空，表示无法读取或为空
# #         return None, 0, 0, []
# #
# #     df_result = detect_anomalies_in_8cols(df, factor=factor)
# #     total_count = len(df_result)
# #
# #     # 找出异常行
# #     mask_anom = df_result["is_anomaly"] == True
# #     anomalies_count = mask_anom.sum()
# #     if anomalies_count == 0:
# #         return df_result, 0, total_count, []
# #
# #     # 取异常行信息
# #     df_anom = df_result.loc[mask_anom].sort_values("index")  # or sort_values("orig_index")
# #     anomaly_rows_info = []
# #     for _, row in df_anom.iterrows():
# #         anomaly_rows_info.append((
# #             int(row["index"]),  # row["index"]就是pandas默认行号; 若要用orig_index则看情况
# #             row["t"],
# #             row["reason"]
# #         ))
# #     return df_result, anomalies_count, total_count, anomaly_rows_info
# #
# #
# # def main():
# #     root_dir = "/media/robert/4TB-SSD/watchped_dataset/Sensor_acc补充"
# #
# #     # 遍历每个子文件夹
# #     for folder_name in os.listdir(root_dir):
# #         folder_path = os.path.join(root_dir, folder_name)
# #         # 只处理形如 WEAR_ 开头的文件夹
# #         if not (os.path.isdir(folder_path) and folder_name.startswith("WEAR_")):
# #             continue
# #
# #         acc_abs_path = os.path.join(folder_path, "WEAR_ACC_ABSOLUTE_01.xlsx")
# #         if not os.path.exists(acc_abs_path):
# #             # 子文件夹没有这个文件则跳过
# #             continue
# #
# #         df_result, anom_count, total_count, anom_info = check_acc_excel(acc_abs_path, factor=10.0)
# #         if df_result is None:
# #             print(f"[警告] {acc_abs_path}: 文件为空或读取失败")
# #             continue
# #
# #         if anom_count == 0:
# #             print(f"[信息] {acc_abs_path}: 无异常 (共 {total_count} 行)")
# #         else:
# #             print(f"[警告] {acc_abs_path}: 发现 {anom_count} 行异常 (共 {total_count} 行)")
# #             max_show = 5
# #             print("  异常行详情(前5条):")
# #             for i, (row_idx, t_val, reason) in enumerate(anom_info):
# #                 if i >= max_show:
# #                     break
# #                 print(f"    -> 原行号:{row_idx}, t={t_val}, reason={reason}")
# #
# #     print("=== 检查完毕 ===")
# #
# #
# # if __name__ == "__main__":
# #     main()
#
# import os
# import pandas as pd
# import numpy as np
#
#
# def load_acc_excel_8cols(excel_path):
#     """
#     从 excel_path 读取 Excel，仅保留 8 列:
#       [t, x, y, z, a, absolute_time, absolute_time_str, formatted_time]
#     若缺列则自动填 NaN，多余列则忽略。
#     返回一个 DataFrame（可能为空）。
#     """
#     wanted_cols = [
#         "t", "x", "y", "z", "a",
#         "absolute_time", "absolute_time_str", "formatted_time"
#     ]
#     try:
#         df_raw = pd.read_excel(excel_path)
#     except Exception as e:
#         print(f"[异常] 无法读取 Excel 文件: {excel_path} - {e}")
#         return pd.DataFrame()  # 返回空表
#
#     # 重排并只保留这 8 列
#     df = df_raw.reindex(columns=wanted_cols)
#     return df
#
#
# def detect_anomalies_in_8cols(df, factor=10.0):
#     """
#     在只含 8 列(t,x,y,z,a,absolute_time,absolute_time_str,formatted_time)的 DataFrame 上，
#     做简单异常检测:
#       - 空白列：只要这 8 列里任一列为 NaN，该行标记异常
#       - 数值偏差太大：对 t 做差分 dt，若 |dt| > factor * median(|dt|)，则标记异常
#         * 特别地，我们跳过第 0 行或手动设置第 0 行 large_diff=False
#
#     返回一个带有以下新增列的 DataFrame：
#       - orig_index: 原始行号
#       - missing_col: bool, 是否空白列
#       - dt: float, t 的差分
#       - large_diff: bool, 是否差分过大
#       - is_anomaly: bool, 是否异常
#       - reason: str, 异常原因
#     """
#     # 给每行加一个 orig_index 来记录原始行号
#     df = df.copy().reset_index(drop=False)
#     df.rename(columns={"index": "orig_index"}, inplace=True)
#
#     needed_cols = ["t", "x", "y", "z", "a", "absolute_time", "absolute_time_str", "formatted_time"]
#
#     # 1) 判定空白列（这 8 列任意为 NaN）
#     # 如果某列在 df 中还不存在，则先补上
#     for c in needed_cols:
#         if c not in df.columns:
#             df[c] = np.nan
#     df["missing_col"] = df[needed_cols].isnull().any(axis=1)
#
#     # 若缺少 t 列，则全部标记异常
#     if "t" not in df.columns:
#         df["dt"] = np.nan
#         df["large_diff"] = True
#         df["is_anomaly"] = True
#         df["reason"] = "缺少 t 列"
#         return df
#
#     # 2) 数值偏差判断
#     #   先根据 t 排序(可能想让其有序再做差分),
#     #   但要保留原行顺序 => 我们只需要差分时有序即可
#     #   这里可以直接对原顺序做 diff，也可以 sort_values("t") 做 diff
#     #   视情况而定。以下采用对原顺序做 diff:
#     #   df.sort_values("orig_index") 保证跟原行顺序一致
#     df = df.sort_values("orig_index", ignore_index=True)
#
#     # 先转 float, 避免异常
#     df["t"] = pd.to_numeric(df["t"], errors="coerce")
#     df["dt"] = df["t"].diff()
#
#     # 第 0 行的 dt = t(1)-t(0)，避免 NaN
#     if len(df) > 1:
#         df.loc[0, "dt"] = df.loc[1, "t"] - df.loc[0, "t"]
#     else:
#         # 只有 1 行，就没法做 diff
#         df["dt"] = 0
#
#     # 计算典型差值 median(|dt|) 跳过第 0 行
#     if len(df) > 1:
#         typical_dt = df["dt"].iloc[1:].abs().median()
#     else:
#         typical_dt = 0
#
#     df["large_diff"] = False
#     if typical_dt > 0:
#         # 标记 large_diff
#         df["large_diff"] = df["dt"].abs() > (factor * typical_dt)
#
#     # *特别需求*: 忽略第 0 行的“数值偏差太大”错误
#     if len(df) > 0:
#         df.loc[0, "large_diff"] = False
#
#     # 3) 组合结果
#     df["is_anomaly"] = df["missing_col"] | df["large_diff"]
#
#     def build_reason(row):
#         reasons = []
#         if row["missing_col"]:
#             reasons.append("存在空白列")
#         if row["large_diff"]:
#             reasons.append("数值偏差太大")
#         return ";".join(reasons)
#
#     df["reason"] = df.apply(build_reason, axis=1)
#
#     return df
#
#
# def check_acc_excel(excel_path, factor=10.0):
#     """
#     读取 excel_path 的 WEAR_ACC_ABSOLUTE_01.xlsx (只保留 8 列)，
#     检测异常并返回:
#       - df_result: 带标记列的 DataFrame
#       - anom_count: 异常行数
#       - total_count: 总行数
#       - anom_info: 列表 [ (orig_index, t, reason), ... ] (已按 orig_index 升序)
#     """
#     df_8 = load_acc_excel_8cols(excel_path)
#     if df_8.empty:
#         return None, 0, 0, []
#
#     df_result = detect_anomalies_in_8cols(df_8, factor=factor)
#     total_count = len(df_result)
#     if total_count == 0:
#         return df_result, 0, 0, []
#
#     # 找出异常行
#     mask_anom = df_result["is_anomaly"]
#     df_anom = df_result.loc[mask_anom].copy()
#
#     # 按原行号排序
#     df_anom = df_anom.sort_values("orig_index")
#
#     anom_count = len(df_anom)
#     anom_info = []
#     for _, row in df_anom.iterrows():
#         # orig_index, t, reason
#         anom_info.append((
#             int(row["orig_index"]),
#             row["t"],
#             row["reason"]
#         ))
#     return df_result, anom_count, total_count, anom_info
#
#
# def main():
#     root_dir = "/media/robert/4TB-SSD/watchped_dataset/Sensor_acc补充"
#
#     # 统计
#     total_checked = 0
#
#     # 遍历子文件夹
#     for folder_name in os.listdir(root_dir):
#         folder_path = os.path.join(root_dir, folder_name)
#         if not os.path.isdir(folder_path):
#             continue
#
#         # 查找 "WEAR_ACC_ABSOLUTE_01.xlsx"
#         acc_abs_path = os.path.join(folder_path, "WEAR_ACC_ABSOLUTE_01.xlsx")
#         if not os.path.exists(acc_abs_path):
#             # 不存在则跳过
#             continue
#
#         total_checked += 1
#         # 检查
#         df_result, anom_count, total_count, anom_info = check_acc_excel(acc_abs_path, factor=10.0)
#         if df_result is None:
#             # 读取失败
#             continue
#         if total_count == 0:
#             # 空表
#             print(f"[信息] {acc_abs_path}: 空文件或无数据")
#             continue
#
#         if anom_count == 0:
#             print(f"[信息] {acc_abs_path}: 无异常 (共 {total_count} 行)")
#         else:
#             print(f"[警告] {acc_abs_path}: 发现 {anom_count} 行异常 (共 {total_count} 行)")
#             # 列举前 5 条
#             print("  异常行详情(前5条):")
#             for i, (orig_idx, t_val, reason) in enumerate(anom_info[:5]):
#                 print(f"    -> 原行号:{orig_idx}, t={t_val}, reason={reason}")
#
#     print("\n=== 检查完毕 ===")
#     print(f"共检查到包含 WEAR_ACC_ABSOLUTE_01.xlsx 的子文件夹数: {total_checked}")
#
#
# if __name__ == "__main__":
#     main()

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
         - 若最前行或最后行只有一侧可比，则只要这一侧也 > threshold，就可视为大偏差。（也可自定义。）
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

    needed_cols = ["t","x","y","z","a","absolute_time","absolute_time_str","formatted_time"]
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
    # 若行 i 两侧都有值，则需 dt_prev[i]>threshold AND dt_next[i]>threshold
    # 若只存在一侧 (i=0 或 i=n-1)，则只要该侧 > threshold 即判定大偏差
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
            # 同时存在
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

        # 只检查 "WEAR_ACC_ABSOLUTE_01.xlsx"
        acc_abs_path = os.path.join(folder_path, "WEAR_ACC_ABSOLUTE_01.xlsx")
        if not os.path.exists(acc_abs_path):
            continue

        total_checked += 1
        df_result, anom_count, total_count, anom_info = check_acc_excel_two_sided(acc_abs_path, factor=10.0)
        if df_result is None:
            # 读取失败
            continue
        if total_count == 0:
            print(f"[信息] {acc_abs_path}: 空或无数据")
            continue

        if anom_count == 0:
            print(f"[信息] {acc_abs_path}: 无异常 (共 {total_count} 行)")
        else:
            print(f"[警告] {acc_abs_path}: 发现 {anom_count} 行异常 (共 {total_count} 行)")
            print("  异常行详情(前5条):")
            for (orig_idx, t_val, reason) in anom_info[:5]:
                print(f"    -> 原行号:{orig_idx}, t={t_val}, reason={reason}")

    print("\n=== 检查完毕 ===")
    print(f"共检查到包含 WEAR_ACC_ABSOLUTE_01.xlsx 的子文件夹数: {total_checked}")

if __name__ == "__main__":
    main()

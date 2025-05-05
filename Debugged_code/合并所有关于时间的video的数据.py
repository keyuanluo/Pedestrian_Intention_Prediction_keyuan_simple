import os
import re
import pandas as pd

def get_video_index(video_id_str):
    """
    从形如 'video_0001' 的字符串中提取数字部分并转为 int，
    以便后续排序时按照数值大小排序。
    例如 'video_0002' -> 2, 'video_0010' -> 10
    """
    match = re.search(r"(\d+)$", video_id_str)
    if match:
        return int(match.group(1))
    else:
        # 若不匹配，给个大值或 0 以防出错
        return 999999

def main():
    # 存放所有 video_XXXX.xlsx 的目录
    in_root = "/media/robert/4TB-SSD/watchped_dataset/Excel_image_时间_video_04"
    # 合并后的输出路径
    out_excel = "/media/robert/4TB-SSD/watchped_dataset/combined_video_image_time_fixed.xlsx"

    all_dfs = []

    for fname in os.listdir(in_root):
        if not fname.lower().endswith(".xlsx"):
            continue  # 只处理 .xlsx 文件

        # 假设文件名形如 "video_0001.xlsx"
        video_id, _ = os.path.splitext(fname)  # => "video_0001"
        fpath = os.path.join(in_root, fname)

        try:
            df = pd.read_excel(fpath)
        except Exception as e:
            print(f"[警告] 无法读取 {fpath}: {e}")
            continue

        if df.empty:
            print(f"[信息] {fpath} 内容为空，跳过。")
            continue

        # 假设已有列: ["image_name", "image_original_time", "image_formatted_time"]
        needed_cols = {"image_name", "image_original_time", "image_formatted_time"}
        real_cols = set(df.columns)
        if not needed_cols.issubset(real_cols):
            print(f"[警告] 文件 {fpath} 缺少列 {needed_cols}，实际列: {real_cols}，跳过。")
            continue

        # 添加一列 video_id
        df["video_id"] = video_id

        # 只保留并按顺序排列需要的 4 列
        df = df[["video_id", "image_name", "image_original_time", "image_formatted_time"]]

        # 1) 修正 image_name 为 6 位零填充
        #    如果原本是数字或短字符串，先转成 int 再 zfill(6)
        #    若不确定数据类型，可做更安全判断
        def fix_image_name(x):
            try:
                return str(int(x)).zfill(6)
            except:
                # 万一解析失败，就当字符串处理再零填充
                return str(x).zfill(6)

        df["image_name"] = df["image_name"].apply(fix_image_name)

        all_dfs.append(df)

    if not all_dfs:
        print("[警告] 未找到可合并的数据文件，未生成大表格。")
        return

    # 合并
    df_all = pd.concat(all_dfs, ignore_index=True)

    # 2) 根据 video_id 中的数字部分进行排序
    df_all["video_index"] = df_all["video_id"].apply(get_video_index)
    # 先按 video_index，再按 image_name 排序
    df_all.sort_values(["video_index", "image_name"], inplace=True)

    # 丢弃临时列 video_index
    df_all.drop(columns=["video_index"], inplace=True)

    # 输出
    df_all.to_excel(out_excel, index=False)
    print(f"[信息] 已合并 {len(all_dfs)} 个文件，总行数 {len(df_all)}，输出到 {out_excel}")

if __name__ == "__main__":
    main()

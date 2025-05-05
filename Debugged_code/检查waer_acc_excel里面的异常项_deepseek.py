import os
import pandas as pd
import numpy as np
from datetime import datetime


def detect_anomalies(root_dir, output_log, time_jump_threshold=1e9):  # 默认时间跳变阈值为1秒（1e9纳秒）
    """检测数据异常并生成详细报告"""
    log = []

    # 遍历所有子文件夹
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        excel_file = os.path.join(folder_path, "WEAR_ACC_ABSOLUTE_01.xlsx")

        if not os.path.exists(excel_file):
            log.append(f"文件不存在: {excel_file}")
            continue

        try:
            # 读取Excel数据
            df = pd.read_excel(excel_file)

            # 基础校验
            if 't' not in df.columns:
                log.append(f"文件结构异常: {excel_file} 缺少t列")
                continue

            anomalies = []
            t_values = pd.to_numeric(df['t'], errors='coerce')

            # 异常检测1：时间戳跳变检测
            for i in range(1, len(t_values) - 1):
                current_t = t_values[i]
                prev_t = t_values[i - 1]
                next_t = t_values[i + 1]

                # 计算前后差值
                diff_prev = abs(current_t - prev_t)
                diff_next = abs(next_t - current_t)

                # 标记异常条件
                if diff_prev > time_jump_threshold and diff_next > time_jump_threshold:
                    anomalies.append({
                        'row': i + 2,  # Excel行号从2开始
                        'type': '时间戳跳变',
                        'value': current_t,
                        'context': f"前值: {prev_t} | 后值: {next_t}",
                        'details': f"异常跳变: 与前后数据差值分别为 {diff_prev} 和 {diff_next} 纳秒"
                    })

            # 异常检测2：空值检测
            null_rows = df[df.isnull().any(axis=1)]
            for idx, row in null_rows.iterrows():
                missing_cols = df.columns[row.isnull()].tolist()
                anomalies.append({
                    'row': idx + 2,
                    'type': '空值异常',
                    'value': None,
                    'context': f"空值列: {', '.join(missing_cols)}",
                    'details': f"第 {idx + 2} 行存在空值"
                })

            # 记录结果
            if anomalies:
                log.append(f"\n=== 在 {excel_file} 中发现 {len(anomalies)} 个异常 ===")
                for anomaly in anomalies:
                    log.append(
                        f"行号: {anomaly['row']}\n"
                        f"类型: {anomaly['type']}\n"
                        f"异常值: {anomaly['value']}\n"
                        f"上下文: {anomaly['context']}\n"
                        f"详情: {anomaly['details']}\n"
                        f"{'-' * 50}"
                    )

        except Exception as e:
            log.append(f"处理文件 {excel_file} 时出错: {str(e)}")

    # 保存检测结果
    with open(output_log, 'w', encoding='utf-8') as f:
        f.write("\n".join(log))

    print(f"检测完成，结果已保存至: {output_log}")


# 使用示例
if __name__ == "__main__":
    # 配置参数
    data_dir = "/media/robert/4TB-SSD/watchped_dataset/Sensor_acc补充"
    log_path = "/media/robert/4TB-SSD/数据质量检测报告.log"

    # 执行检测（时间跳变阈值设置为1秒=1e9纳秒）
    detect_anomalies(data_dir, log_path, time_jump_threshold=1e9)
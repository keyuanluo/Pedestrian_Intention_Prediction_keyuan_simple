import os

def check_second_line_is_complete(acc_file):
    """
    打开 acc_file，读取前两行：
      - 第 1 行：表头 (t x y z a)
      - 第 2 行：第一条数据行
    如果第二行存在并至少有 5 个字段，则返回 True；否则返回 False。
    """

    if not os.path.exists(acc_file):
        print(f"[异常] 文件不存在: {acc_file}")
        return False

    try:
        with open(acc_file, 'r', encoding='utf-8', errors='replace') as f:
            header_line = f.readline().rstrip('\n')  # 读取表头行
            second_line = f.readline().rstrip('\n')  # 读取第二行(第一条数据)
    except Exception as e:
        print(f"[异常] 无法读取文件: {acc_file} - {e}")
        return False

    # 如果第二行为空，视为不完整
    if not second_line.strip():
        print(f"[异常] {acc_file} 第二行(第一条数据)为空")
        return False

    # 智能检测分隔符：分号 ; 、制表符 \t、或空白
    if ';' in second_line:
        fields = second_line.split(';')
    elif '\t' in second_line:
        fields = second_line.split('\t')
    else:
        # 默认用空白分隔
        fields = second_line.split()

    if len(fields) < 5:
        print(f"[异常] {acc_file} 第二行字段不足 5 个: {fields}")
        return False

    # 若需要更严格的校验(如匹配某些列值类型), 可在此处再加逻辑
    return True

def main():
    root_dir = "/media/robert/4TB-SSD/watchped_dataset/Sensor_acc补充"

    # 汇总结果
    results = []

    # 遍历子文件夹
    for folder_name in os.listdir(root_dir):
        # 只处理 WEAR_ 开头的文件夹
        if not folder_name.startswith("WEAR_"):
            continue

        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        acc_file = os.path.join(folder_path, "WEAR_ACC.csv")
        if os.path.exists(acc_file):
            is_ok = check_second_line_is_complete(acc_file)
            if is_ok:
                print(f"[信息] {acc_file} 第二行数据完整")
                results.append((acc_file, "OK"))
            else:
                results.append((acc_file, "Incomplete"))
        else:
            # 如果需要也可记录, 这里跳过
            pass

    # 统计异常数量
    error_count = sum(1 for _, status in results if status == "Incomplete")

    print("\n=== 检查结果汇总 ===")
    for path, status in results:
        print(f"{path} -> {status}")

    # 如果没有异常则打印 "全部合格"，否则打印异常数量
    if error_count == 0:
        print("\n全部合格！")
    else:
        print(f"\n共有 {error_count} 个不合格")

if __name__ == "__main__":
    main()

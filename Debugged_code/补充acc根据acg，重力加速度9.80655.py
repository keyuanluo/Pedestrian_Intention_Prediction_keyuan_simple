# import os
# import pandas as pd
#
#
# def main():
#     root_dir = "/media/robert/4TB-SSD/watchped_dataset/Sensor_video_acc补充"
#     gravity_value = 9.80655  # 要减去的重力加速度
#
#     for folder_name in os.listdir(root_dir):
#         folder_path = os.path.join(root_dir, folder_name)
#         if not os.path.isdir(folder_path):
#             continue  # 只处理文件夹
#
#         # 构造路径
#         acc_file = os.path.join(folder_path, "WEAR_ACC.csv")
#         acg_file = os.path.join(folder_path, "WEAR_ACG.csv")
#
#         # 1) 判断 WEAR_ACC.csv 是否缺失
#         if not os.path.exists(acc_file):
#             print(f"[{folder_name}] 缺失 WEAR_ACC.csv")
#
#             # 2) 检查 WEAR_ACG.csv 是否存在
#             if os.path.exists(acg_file):
#                 print(f"  -> 准备从 WEAR_ACG.csv 生成新的 WEAR_ACC.csv")
#
#                 # 3) 读取 WEAR_ACG.csv
#                 #    假设原文件是空格/制表符分隔，并有 5 列: t x y z a
#                 #    如果实际是逗号分隔, 可改 sep=',' 等
#                 #    如果首行无列名, 需指定 names=['t','x','y','z','a']
#                 df_acg = pd.read_csv(acg_file, sep=r'\s+', header=0, names=['t', 'x', 'y', 'z', 'a'])
#
#                 # 4) 对 z 列减去 9.80655
#                 df_acg['z'] = df_acg['z'] - gravity_value
#
#                 # 5) 将结果保存为 WEAR_ACC.csv
#                 #    保持同样的 5 列: t x y z a
#                 #    sep=' ' 表示以空格分隔
#                 df_acg.to_csv(acc_file, sep=' ', index=False, header=False)
#                 print(f"  -> 已生成新的 WEAR_ACC.csv 到 {acc_file}")
#             else:
#                 print(f"  -> 同时缺失 WEAR_ACG.csv，无法生成 WEAR_ACC.csv")
#         else:
#             # 如果已存在，则无需处理
#             # 你也可以选择检查内容或覆盖之类
#             print(f"[{folder_name}] 已有 WEAR_ACC.csv，无需补充")
#
#
# if __name__ == "__main__":
#     main()

import os
import pandas as pd

def main():
    root_dir = "/media/robert/4TB-SSD/watchped_dataset/Sensor_video_acc补充"
    gravity_value = 9.80655  # 要减去的重力加速度

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue  # 只处理文件夹

        acc_file = os.path.join(folder_path, "WEAR_ACC.csv")
        acg_file = os.path.join(folder_path, "WEAR_ACG.csv")

        # 1) 判断 WEAR_ACC.csv 是否缺失
        if not os.path.exists(acc_file):
            print(f"[{folder_name}] 缺失 WEAR_ACC.csv")

            # 2) 检查 WEAR_ACG.csv 是否存在
            if os.path.exists(acg_file):
                print(f"  -> 准备从 WEAR_ACG.csv 生成新的 WEAR_ACC.csv")

                try:
                    # 3) 读取 WEAR_ACG.csv
                    #    若文件是空格/制表符分隔且无表头，可这样：
                    df_acg = pd.read_csv(
                        acg_file,
                        delim_whitespace=True, # 如果是空白分隔
                        header=None,           # 文件无表头
                        names=['t','x','y','z','a']  # 指定5列
                    )

                    # 若实际是逗号分隔(CSV)且无表头, 则改成:
                    # df_acg = pd.read_csv(acg_file, sep=',', header=None, names=['t','x','y','z','a'])

                    # 4) 对 z 列减去 9.80655
                    df_acg['z'] = df_acg['z'] - gravity_value

                    # 5) 将结果保存为 WEAR_ACC.csv (5列: t x y z a)
                    df_acg.to_csv(acc_file, sep=' ', index=False, header=False)
                    print(f"  -> 已生成新的 WEAR_ACC.csv 到 {acc_file}")

                except Exception as e:
                    print(f"  -> 读取/生成出错: {e}")
            else:
                print(f"  -> 同时缺失 WEAR_ACG.csv，无法生成 WEAR_ACC.csv")
        else:
            # 如果已存在，则无需处理
            print(f"[{folder_name}] 已有 WEAR_ACC.csv，无需补充")

if __name__ == "__main__":
    main()

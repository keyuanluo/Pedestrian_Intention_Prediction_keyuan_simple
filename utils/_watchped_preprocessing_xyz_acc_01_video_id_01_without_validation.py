import torch
import os
import numpy as np
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_all(seed):
    torch.cuda.empty_cache()
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def binary_acc(label, pred):
    """
    计算二分类准确率:
      label, pred 的形状均为 [batch_size, 1]
    """
    label_tag = torch.round(label)
    correct_results_sum = (label_tag == pred).sum().float()
    # 这里除以 batch_size
    acc = correct_results_sum / pred.shape[0]
    return acc


def end_point_loss(reg_criterion, pred, end_point):
    """
    将预测和真实 endpoint 的归一化坐标还原到原图尺寸后计算 MSE 等回归损失
    """
    for i in range(4):
        if i == 0 or i == 2:
            pred[:, i] = pred[:, i] * 1920
            end_point[:, i] = end_point[:, i] * 1920
        else:
            pred[:, i] = pred[:, i] * 1080
            end_point[:, i] = end_point[:, i] * 1080
    return reg_criterion(pred, end_point)


def train(model, train_loader, class_criterion, reg_criterion, optimizer,
          checkpoint_filepath, writer, args):
    """
    训练函数（修改为无验证集）：
      - DataLoader 返回 5 个元素: (bbox, label, vel, traj, acc)
      - 采用训练集上的指标作为性能提升标准
    """
    best_train_loss = np.inf
    num_steps_wo_improvement = 0
    save_times = 0
    epochs = args.epochs

    # 如果是调试模式，可减少 epochs
    if args.learn:
        epochs = 5

    time_crop = args.time_crop

    for epoch in range(epochs):
        nb_batches_train = len(train_loader)
        # 用于累加训练集的准确率
        train_acc_score = 0.0

        model.train()
        f_losses = 0.0
        cls_losses = 0.0
        reg_losses = 0.0

        print(f'Epoch: {epoch + 1} training...')

        for bbox, label, vel, traj, acc_data in train_loader:
            # label, bbox, vel, acc_data 均送到 GPU
            label = label.reshape(-1, 1).to(device).float()
            bbox = bbox.to(device)
            vel = vel.to(device)
            acc_data = acc_data.to(device)

            # 取轨迹最后一帧作为 endpoint
            end_point = traj.to(device)[:, -1, :]

            # 随机裁剪时序片段
            if np.random.randint(10) >= 5 and time_crop:
                crop_size = np.random.randint(args.sta_f, args.end_f)
                bbox = bbox[:, -crop_size:, :]
                vel = vel[:, -crop_size:, :]
                acc_data = acc_data[:, -crop_size:, :]

            # 调用模型 forward，传入加速度 acc_data
            pred, point, s_cls, s_reg = model(bbox, vel, acc_data)

            cls_loss = class_criterion(pred, label)
            reg_loss = reg_criterion(point, end_point)
            # 自适应多任务损失
            f_loss = cls_loss / (s_cls * s_cls) + reg_loss / (s_reg * s_reg) + torch.log(s_cls) + torch.log(s_reg)

            model.zero_grad()
            f_loss.backward()
            optimizer.step()

            # 记录损失
            f_losses += f_loss.item()
            cls_losses += cls_loss.item()
            reg_losses += reg_loss.item()

            # 计算并累加准确率
            batch_acc = binary_acc(label, torch.round(pred))
            train_acc_score += batch_acc

        # 写入 TensorBoard
        writer.add_scalar('training full_loss', f_losses / nb_batches_train, epoch + 1)
        writer.add_scalar('training cls_loss', cls_losses / nb_batches_train, epoch + 1)
        writer.add_scalar('training reg_loss', reg_losses / nb_batches_train, epoch + 1)
        writer.add_scalar('training Acc', train_acc_score / nb_batches_train, epoch + 1)

        print(f"Epoch {epoch + 1}: "
              f"| Train_Loss {f_losses / nb_batches_train:.4f} "
              f"| Train Cls_loss {cls_losses / nb_batches_train:.4f} "
              f"| Train Reg_loss {reg_losses / nb_batches_train:.4f} "
              f"| Train_Acc {train_acc_score / nb_batches_train:.4f} ")

        # 这里使用训练集的分类损失作为性能提升指标（无验证集时）
        if best_train_loss > cls_losses / nb_batches_train:
            best_train_loss = cls_losses / nb_batches_train
            num_steps_wo_improvement = 0
            save_times += 1
            print(f"{save_times} time(s) File saved.\n")

            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'Accuracy': train_acc_score / nb_batches_train,
                'LOSS': f_losses / nb_batches_train,
            }, checkpoint_filepath)
            print('Update improvement.\n')
        else:
            num_steps_wo_improvement += 1
            print(f"{num_steps_wo_improvement} time(s) no improvement.\n")

        if num_steps_wo_improvement == 3000:
            print(f"Early stopping on epoch: {epoch + 1}")
            break

    print(f'save file times: {save_times}.\n')


def test(model, test_data):
    """
    测试函数：
      - 解包数据：(bbox, label, vel, traj, acc_data, vid_ids)
      - 预测后返回：preds, labels, video_id_list
      - 同时打印出预测错误的样本信息（包括 video_id）
    """
    print("Testing...")
    with torch.no_grad():
        model.eval()
        video_id_list = []
        preds = None
        labels = None

        for step, (bbox, label, vel, traj, acc_data, vid_ids) in enumerate(test_data):
            # 数据预处理：调整 label 形状并送入设备
            label = label.reshape(-1, 1).to(device).float()
            bbox = bbox.to(device)
            vel = vel.to(device)
            acc_data = acc_data.to(device)

            # 模型预测
            pred, _, _, _ = model(bbox, vel, acc_data)
            video_id_list.extend(vid_ids)

            # 将每个 batch 的预测和标签拼接起来
            if step == 0:
                preds = pred
                labels = label
            else:
                preds = torch.cat((preds, pred), dim=0)
                labels = torch.cat((labels, label), dim=0)

        if len(video_id_list) != preds.shape[0]:
            print("Warning: video_id_list 的长度与预测样本数量不一致！")

        preds_rounded = torch.round(preds)
        incorrect_indices = (preds_rounded != labels).nonzero(as_tuple=True)[0]

        if len(incorrect_indices) == 0:
            print("所有样本均预测正确。")
        else:
            print("\n以下样本预测错误：")
            for idx in incorrect_indices:
                vid_str = video_id_list[idx]
                print(f"Sample index={idx}, video_id={vid_str}, pred={preds_rounded[idx].item()}, label={labels[idx].item()}")

    print("Testing completed.")
    return preds, labels, video_id_list


def balance_dataset(dataset, flip=True):
    """
    数据平衡处理:
      - 若存在 'acc' 字段, 一并进行镜像翻转与抽样删除
    """
    d = {
        'bbox': dataset['bbox'].copy(),
        'pid': dataset['pid'].copy(),
        'activities': dataset['activities'].copy(),
        'image': dataset['image'].copy(),
        'center': dataset['center'].copy(),
        'vehicle_act': dataset['vehicle_act'].copy(),
        'image_dimension': (1920, 1080)
    }

    # 如果原数据集里有 'acc' 字段，就拷贝过去
    if 'acc' in dataset:
        d['acc'] = dataset['acc'].copy()

    gt_labels = [gt[0] for gt in d['activities']]
    num_pos_samples = np.count_nonzero(np.array(gt_labels))
    num_neg_samples = len(gt_labels) - num_pos_samples

    if num_neg_samples == num_pos_samples:
        print('Positive samples is equal to negative samples.')
    else:
        print(f'Unbalanced: \t Postive: {num_pos_samples} \t Negative: {num_neg_samples}')
        if num_neg_samples > num_pos_samples:
            gt_augment = 1
        else:
            gt_augment = 0

        img_width = d['image_dimension'][0]
        num_samples = len(d['pid'])

        # 镜像翻转标签=gt_augment的样本
        for i in range(num_samples):
            if d['activities'][i][0][0] == gt_augment:
                # center
                flipped_center = d['center'][i].copy()
                flipped_center = [[img_width - c[0], c[1]] for c in flipped_center]
                d['center'].append(flipped_center)

                # bbox
                flipped_bbox = d['bbox'][i].copy()
                flipped_bbox = [
                    np.array([img_width - c[2], c[1], img_width - c[0], c[3]])
                    for c in flipped_bbox
                ]
                d['bbox'].append(flipped_bbox)

                d['pid'].append(d['pid'][i].copy())
                d['activities'].append(d['activities'][i].copy())
                d['vehicle_act'].append(d['vehicle_act'][i].copy())

                flipped_image = d['image'][i].copy()
                flipped_image = [c.replace('.png', '_flip.png') for c in flipped_image]
                d['image'].append(flipped_image)

                # 同步镜像加速度(若需要翻转加速度,可在此处改值,否则只做复制)
                if 'acc' in d:
                    d['acc'].append(d['acc'][i].copy())

        # 重新统计并随机删除多余的样本
        gt_labels = [gt[0] for gt in d['activities']]
        num_pos_samples = np.count_nonzero(np.array(gt_labels))
        num_neg_samples = len(gt_labels) - num_pos_samples

        if num_neg_samples > num_pos_samples:
            rm_index = np.where(np.array(gt_labels) == 0)[0]
        else:
            rm_index = np.where(np.array(gt_labels) == 1)[0]

        dif_samples = abs(num_neg_samples - num_pos_samples)
        np.random.seed(42)
        np.random.shuffle(rm_index)
        rm_index = rm_index[:dif_samples]

        for k in d.keys():
            seq_data_k = d[k]
            if isinstance(seq_data_k, list):
                d[k] = [seq_data_k[i] for i in range(len(seq_data_k)) if i not in rm_index]

        new_gt_labels = [gt[0] for gt in d['activities']]
        num_pos_samples = np.count_nonzero(np.array(new_gt_labels))
        print(f'Balanced: Postive: {num_pos_samples} '
              f'\t Negative: {len(d["activities"]) - num_pos_samples}\n')
        print(f'Total Number of samples: {len(d["activities"])}\n')

    return d


def tte_dataset(dataset, time_to_event, overlap, obs_length):
    """
    同步对 'acc' 字段进行裁剪
    """
    d_obs = {
        'bbox': dataset['bbox'].copy(),
        'pid': dataset['pid'].copy(),
        'activities': dataset['activities'].copy(),
        'image': dataset['image'].copy(),
        'vehicle_act': dataset['vehicle_act'].copy(),
        'center': dataset['center'].copy()
    }

    d_tte = {
        'bbox': dataset['bbox'].copy(),
        'pid': dataset['pid'].copy(),
        'activities': dataset['activities'].copy(),
        'image': dataset['image'].copy(),
        'vehicle_act': dataset['vehicle_act'].copy(),
        'center': dataset['center'].copy()
    }

    # 如果有 acc, 也一起复制
    if 'acc' in dataset:
        d_obs['acc'] = dataset['acc'].copy()
        d_tte['acc'] = dataset['acc'].copy()

    if isinstance(time_to_event, int):
        # 简单模式: 直接取后 obs_length + time_to_event 帧
        for k in d_obs.keys():
            if k in ['bbox','pid','activities','image','vehicle_act','center','acc']:
                for i in range(len(d_obs[k])):
                    d_obs[k][i] = d_obs[k][i][- obs_length - time_to_event : -time_to_event]
                    d_tte[k][i] = d_tte[k][i][- time_to_event :]
        d_obs['tte'] = [[time_to_event]] * len(dataset['bbox'])
        d_tte['tte'] = [[time_to_event]] * len(dataset['bbox'])
    else:
        # time_to_event 为 [30,60] 等区间, 处理更复杂
        olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length)
        olap_res = 1 if olap_res < 1 else olap_res

        for k in d_obs.keys():
            if k in ['bbox','pid','activities','image','vehicle_act','center','acc']:
                seqs_obs = []
                seqs_tte = []
                old_data = d_obs[k]
                for seq in old_data:
                    start_idx = len(seq) - obs_length - time_to_event[1]
                    end_idx = len(seq) - obs_length - time_to_event[0]
                    for i in range(start_idx, end_idx, olap_res):
                        seqs_obs.append(seq[i : i + obs_length])
                        seqs_tte.append(seq[i + obs_length : ])
                d_obs[k] = seqs_obs
                d_tte[k] = seqs_tte

        # 同步生成 tte_seq
        tte_seq = []
        for seq in dataset['bbox']:
            start_idx = len(seq) - obs_length - time_to_event[1]
            end_idx = len(seq) - obs_length - time_to_event[0]
            tte_seq.extend([[len(seq) - (i + obs_length)] for i in range(start_idx, end_idx, olap_res)])
        d_obs['tte'] = tte_seq.copy()
        d_tte['tte'] = tte_seq.copy()

    # 移除过短序列
    try:
        time_to_event_0 = time_to_event[0]
    except:
        time_to_event_0 = time_to_event

    remove_index = []
    for idx, (seq_obs, seq_tte) in enumerate(zip(d_obs['bbox'], d_tte['bbox'])):
        if len(seq_obs) < 16 or len(seq_tte) < time_to_event_0:
            remove_index.append(idx)

    for k in d_obs.keys():
        for j in sorted(remove_index, reverse=True):
            del d_obs[k][j]
            del d_tte[k][j]

    return d_obs, d_tte


def normalize_bbox(dataset, width=1920, height=1080):
    normalized_set = []
    for sequence in dataset:
        if not sequence:
            continue
        normalized_sequence = []
        for bbox in sequence:
            np_bbox = np.zeros(4)
            np_bbox[0] = bbox[0] / width
            np_bbox[2] = bbox[2] / width
            np_bbox[1] = bbox[1] / height
            np_bbox[3] = bbox[3] / height
            normalized_sequence.append(np_bbox)
        normalized_set.append(np.array(normalized_sequence))
    return normalized_set


def normalize_traj(dataset, width=1920, height=1080):
    normalized_set = []
    for sequence in dataset:
        if not sequence:
            continue
        normalized_sequence = []
        for bbox in sequence:
            np_bbox = np.zeros(4)
            np_bbox[0] = bbox[0]
            np_bbox[2] = bbox[2]
            np_bbox[1] = bbox[1]
            np_bbox[3] = bbox[3]
            normalized_sequence.append(np_bbox)
        normalized_set.append(np.array(normalized_sequence))
    return normalized_set


def prepare_label(dataset):
    """
    将 dataset['activities'] 中的意图标记提取为 0/1
    """
    labels = np.zeros(len(dataset), dtype='int64')
    for step, action in enumerate(dataset):
        if action:
            labels[step] = action[0][0]
    return labels


def pad_sequence(inp_list, max_len):
    """
    对每个序列补零到 max_len 长度
    """
    padded_sequence = []
    for source in inp_list:
        if len(source) == 0:
            continue
        target = np.array([source[0]] * max_len)
        target[-len(source):, :] = source
        padded_sequence.append(target)
    return padded_sequence

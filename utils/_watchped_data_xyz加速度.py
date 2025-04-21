
import sys
import pickle
import cv2

import numpy as np
import xml.etree.ElementTree as ET

from os.path import join, abspath, exists
from os import listdir, makedirs
from sklearn.model_selection import train_test_split, KFold


class WATCHPED(object):
    def __init__(self, data_path='/media/robert/4TB-SSD/pkl运行', regen_pkl=False):
        self._name = 'WATCHPED'
        self._regen_pkl = regen_pkl
        self._image_ext = '.png'

        # Paths
        self._watchped_path = data_path if data_path else self._get_default_path()
        assert exists(self._watchped_path), 'watchped path does not exist: {}'.format(self._watchped_path)
        self._data_split_ids_path = join(self._watchped_path, 'split_ids')
        self._annotation_path = join(self._watchped_path, 'annotations')
        self._annotation_vehicle_path = join(self._watchped_path, 'annotations_vehicle')
        self._annotation_traffic_path = join(self._watchped_path, 'annotations_traffic')
        self._annotation_attributes_path = join(self._watchped_path, 'annotations_attributes')
        self._annotation_appearance_path = join(self._watchped_path, 'annotations_appearance')
        self._clips_path = join(self._watchped_path, 'watchped_clips')
        self._images_path = join(self._watchped_path, 'images')

    @property
    def cache_path(self):
        cache_path = abspath(join(self._watchped_path, 'data_cache'))
        if not exists(cache_path):
            makedirs(cache_path)
        return cache_path

    def _get_default_path(self):
        return 'dataset/watchped'

    def _get_video_ids(self):
        return [vid.split('.')[0] for vid in listdir(self._annotation_path)]

    def update_progress(self, progress):
        barLength = 20
        status = ''
        if isinstance(progress, int):
            progress = float(progress)
        block = int(round(barLength * progress))
        text = '\r[{}] {:0.2f}% {}'.format('#' * block + '-' * (barLength - block), progress * 100, status)
        sys.stdout.write(text)
        sys.stdout.flush()

    def extract_and_save_images(self):
        videos = [f.split('.')[0] for f in sorted(listdir(self._clips_path))]
        for vid in videos:
            path_to_file = join(self._annotation_path, vid + '.xml')
            print(vid)
            tree = ET.parse(path_to_file)
            num_frames = int(tree.find('./meta/task/size').text)
            video_clip_path = join(self._clips_path, vid + '.mp4')
            save_images_path = join(self._images_path, vid)
            if not exists(save_images_path):
                makedirs(save_images_path)
            vidcap = cv2.VideoCapture(video_clip_path)
            success, image = vidcap.read()
            frame_num = 0
            img_count = 0
            if not success:
                print('Failed to open the video {}'.format(vid))
            while success:
                self.update_progress(img_count / num_frames)
                img_count += 1
                img_path = join(save_images_path, '{:05d}.png'.format(frame_num))
                if not exists(img_path):
                    cv2.imwrite(img_path, image)
                success, image = vidcap.read()
                frame_num += 1
            if num_frames != img_count:
                print('num images don\'t match {}/{}'.format(num_frames, img_count))
            print('\n')

    # 一些 map_text_to_scalar、map_scalar_to_text 的函数省略（与你原代码相同）

    def generate_database(self):
        print('------------------------------------------------')
        print('Generating databse for watchped')
        cache_file = join(self.cache_path, 'watchped_10.pkl')
        if exists(cache_file) and not self._regen_pkl:
            with open(cache_file, 'rb') as fid:
                try:
                    database = pickle.load(fid)
                except:
                    database = pickle.load(fid, encoding='bytes')
            print('watchped database loaded from {}'.format(cache_file))
            return database

        # 如果要重新生成数据库的逻辑，这里省略，保留你原本的生成过程
        # ...
        # 最后写入 cache_file
        # ...
        # return database

        # 下方仅示例：若 _regen_pkl=True，才会执行到这里
        # 这里是你原来的组装 database 的代码
        # 省略 ...
        # with open(cache_file, 'wb') as fid:
        #     pickle.dump(database, fid, pickle.HIGHEST_PROTOCOL)
        # print('The database is written to {}'.format(cache_file))
        # return database

    def get_data_stats(self):
        annotations = self.generate_database()
        videos_count = len(annotations.keys())
        # ... 统计逻辑省略

    # balance_samples_count 省略（原样）

    def _get_video_ids_split(self, image_set, subset='default'):
        vid_ids = []
        sets = [image_set] if image_set != 'all' else ['train', 'test', 'val']
        for s in sets:
            vid_id_file = join(self._data_split_ids_path, subset, s + '.txt')
            with open(vid_id_file, 'rt') as fid:
                vid_ids.extend([x.strip() for x in fid.readlines()])
        return vid_ids

    def _get_pedestrian_ids(self, sample_type='all'):
        annotations = self.generate_database()
        pids = []
        for vid in sorted(annotations):
            if sample_type == 'beh':
                pids.extend([p for p in annotations[vid]['ped_annotations'].keys() if 'b' in p])
            else:
                pids.extend(annotations[vid]['ped_annotations'].keys())
        return pids

    def _get_random_pedestrian_ids(self, image_set, ratios=None, val_data=True, regen_data=False, sample_type='all'):
        cache_file = join(self.cache_path, 'random_samples.pkl')
        if exists(cache_file) and not regen_data:
            print('Random smaple currently exists.\n Loading from %s' % cache_file)
            with open(cache_file, 'rb') as fid:
                try:
                    rand_samples = pickle.load(fid)
                except:
                    rand_samples = pickle.load(fid, encoding='bytes')
                # ... 省略原逻辑
            return rand_samples[image_set]

        # 如果需要重新生成随机数据的逻辑
        # ...
        # return sample_split[image_set]

    def _get_kfold_pedestrian_ids(self, image_set, num_folds=5, fold=1, sample_type='all'):
        # ... 省略原逻辑
        pass

    def _get_data_ids(self, image_set, params):
        _pids = None
        if params['data_split_type'] == 'default':
            return self._get_video_ids_split(image_set, params['subset']), _pids
        video_ids = self._get_video_ids_split('all', params['subset'])
        if params['data_split_type'] == 'random':
            params['random_params']['sample_type'] = params['sample_type']
            _pids = self._get_random_pedestrian_ids(image_set, **params['random_params'])
        elif params['data_split_type'] == 'kfold':
            params['kfold_params']['sample_type'] = params['sample_type']
            _pids = self._get_kfold_pedestrian_ids(image_set, **params['kfold_params'])
        return video_ids, _pids

    def _squarify(self, bbox, ratio, img_width):
        width = abs(bbox[0] - bbox[2])
        height = abs(bbox[1] - bbox[3])
        width_change = height * ratio - width
        bbox[0] = bbox[0] - width_change / 2
        bbox[2] = bbox[2] + width_change / 2
        if bbox[0] < 0:
            bbox[0] = 0
        if bbox[2] > img_width:
            bbox[0] = bbox[0] - bbox[2] + img_width
            bbox[2] = img_width
        return bbox

    # get_detection_data 系列函数省略（原样）

    def _print_dict(self, dic):
        for k, v in dic.items():
            print('%s: %s' % (str(k), str(v)))

    def _height_check(self, height_rng, frame_ids, boxes, images, occlusion):
        imgs, box, frames, occ = [], [], [], []
        for i, b in enumerate(boxes):
            bbox_height = abs(b[1] - b[3])
            if height_rng[0] <= bbox_height <= height_rng[1]:
                box.append(b)
                imgs.append(images[i])
                frames.append(frame_ids[i])
                occ.append(occlusion[i])
        return imgs, box, frames, occ

    def _get_center(self, box):
        return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

    def _get_image_path(self, vid, f):
        return join(self._images_path, vid, '{:05d}.png'.format(f))

    def generate_data_trajectory_sequence(self, image_set, **opts):
        params = {
            'fstride': 1,
            'sample_type': 'all',
            'subset': 'default',
            'height_rng': [0, float('inf')],
            'squarify_ratio': 0,
            'data_split_type': 'default',
            'seq_type': 'intention',
            'min_track_size': 15,
            'random_params': {'ratios': None, 'val_data': True, 'regen_data': False},
            'kfold_params': {'num_folds': 5, 'fold': 1}
        }
        params.update(opts)
        print('--------------------------------------------------------')
        print('Generating action sequence data.')
        self._print_dict(params)

        annot_database = self.generate_database()

        if params['seq_type'] == 'trajectory':
            sequence = self._get_trajectories(image_set, annot_database, **params)
        elif params['seq_type'] == 'crossing':
            sequence = self._get_crossing(image_set, annot_database, **params)
        elif params['seq_type'] == 'intention':
            sequence = self._get_intention(image_set, annot_database, **params)
        else:
            raise ValueError(f"Unknown seq_type: {params['seq_type']}")

        return sequence

    # =============== 以下三个函数，加入对 'acc' 的处理 ===================

    def _get_trajectories(self, image_set, annot_database, **params):
        print('---------------------------------------------------------')
        print('Generating trajectory data.')
        num_pedestrians = 0
        seq_stride = params['fstride']
        sq_ratio = params['squarify_ratio']
        height_rng = params['height_rng']

        image_seq, pids_seq = [], []
        box_seq, center_seq, occ_seq = [], [], []
        intent_seq = []
        vehicle_seq = []

        # --- ADDED for acc ---
        acc_seq = []  # 用于存储每个行人的加速度

        video_ids, _pids = self._get_data_ids(image_set, params)
        for vid in sorted(video_ids):
            img_width = annot_database[vid]['width']
            pid_annots = annot_database[vid]['ped_annotations']
            vid_annots = annot_database[vid]['vehicle_annotations']
            for pid in sorted(pid_annots):
                if params['data_split_type'] != 'default' and pid not in _pids:
                    continue
                if 'p' in pid:
                    continue
                if params['sample_type'] == 'beh' and 'b' not in pid:
                    continue

                num_pedestrians += 1
                frame_ids = pid_annots[pid]['frames']
                images = [join(self._watchped_path, 'images', vid, '{:05d}.png'.format(f)) for f in frame_ids]
                boxes = pid_annots[pid]['bbox']
                occlusions = pid_annots[pid]['occlusion']

                # --- ADDED for acc ---
                # 如果 pkl 中有 acc，就取出来；否则给个空列表
                if 'acc' in pid_annots[pid]:
                    all_acc = pid_annots[pid]['acc']
                else:
                    all_acc = []

                # 高度范围过滤
                if height_rng[0] > 0 or height_rng[1] < float('inf'):
                    images, boxes, frame_ids, occlusions = self._height_check(height_rng, frame_ids, boxes, images, occlusions)
                    if all_acc:
                        all_acc = [all_acc[i] for i in range(len(all_acc)) if i < len(frame_ids)]

                # 判断序列长度
                if len(boxes) / seq_stride < params['min_track_size']:
                    continue

                if sq_ratio:
                    boxes = [self._squarify(b, sq_ratio, img_width) for b in boxes]

                ped_ids = [[pid]] * len(boxes)
                # intent
                if 'b' not in pid:
                    intent = [[0]] * len(boxes)
                else:
                    crossing_val = annot_database[vid]['ped_annotations'][pid]['attributes'].get('crossing', -1)
                    if crossing_val == -1:
                        intent = [[0]] * len(boxes)
                    else:
                        intent = [[1]] * len(boxes)

                center = [self._get_center(b) for b in boxes]
                occ_seq.append(occlusions[::seq_stride])
                image_seq.append(images[::seq_stride])
                box_seq.append(boxes[::seq_stride])
                center_seq.append(center[::seq_stride])
                intent_seq.append(intent[::seq_stride])
                pids_seq.append(ped_ids[::seq_stride])
                vehicle_seq.append([[vid_annots[i]] for i in frame_ids][::seq_stride])

                # --- ADDED for acc ---
                if all_acc:
                    # 截断 + 步幅
                    all_acc = all_acc[::seq_stride]
                acc_seq.append(all_acc)

        print('Split: %s' % image_set)
        print('Number of pedestrians: %d ' % num_pedestrians)
        print('Total number of samples: %d ' % len(image_seq))

        return {
            'image': image_seq,
            'pid': pids_seq,
            'bbox': box_seq,
            'center': center_seq,
            'occlusion': occ_seq,
            'vehicle_act': vehicle_seq,
            'intent': intent_seq,
            # --- ADDED for acc ---
            'acc': acc_seq
        }

    def _get_crossing(self, image_set, annot_database, **params):
        print('---------------------------------------------------------')
        print("Generating crossing data")
        num_pedestrians = 0
        seq_stride = params['fstride']
        sq_ratio = params['squarify_ratio']
        height_rng = params['height_rng']

        image_seq, pids_seq = [], []
        box_seq, center_seq, occ_seq = [], [], []
        intent_seq = []
        vehicle_seq = []
        activities = []

        # --- ADDED for acc ---
        acc_seq = []

        video_ids, _pids = self._get_data_ids(image_set, params)
        for vid in sorted(video_ids):
            img_width = annot_database[vid]['width']
            img_height = annot_database[vid]['height']
            pid_annots = annot_database[vid]['ped_annotations']
            vid_annots = annot_database[vid]['vehicle_annotations']

            for pid in sorted(pid_annots):
                if params['data_split_type'] != 'default' and pid not in _pids:
                    continue
                if 'p' in pid:
                    continue
                if params['sample_type'] == 'beh' and 'b' not in pid:
                    continue

                num_pedestrians += 1
                frame_ids = pid_annots[pid]['frames']

                if 'b' in pid:
                    event_frame = pid_annots[pid]['attributes'].get('crossing_point', -1)
                else:
                    event_frame = -1

                if event_frame == -1:
                    end_idx = -3
                else:
                    end_idx = frame_ids.index(event_frame)

                boxes = pid_annots[pid]['bbox'][:end_idx + 1]
                frame_ids = frame_ids[: end_idx + 1]
                images = [self._get_image_path(vid, f) for f in frame_ids]
                occlusions = pid_annots[pid]['occlusion'][:end_idx + 1]

                # --- ADDED for acc ---
                if 'acc' in pid_annots[pid]:
                    all_acc = pid_annots[pid]['acc'][:end_idx + 1]
                else:
                    all_acc = []

                # 高度过滤
                if height_rng[0] > 0 or height_rng[1] < float('inf'):
                    images, boxes, frame_ids, occlusions = self._height_check(height_rng, frame_ids, boxes, images, occlusions)
                    if all_acc:
                        # 同步删掉对应的加速度
                        all_acc = [all_acc[i] for i in range(len(all_acc)) if i < len(images)]

                if len(boxes) / seq_stride < params['min_track_size']:
                    continue

                if sq_ratio:
                    boxes = [self._squarify(b, sq_ratio, img_width) for b in boxes]

                image_seq.append(images[::seq_stride])
                box_seq.append(boxes[::seq_stride])
                center_seq.append([self._get_center(b) for b in boxes][::seq_stride])
                occ_seq.append(occlusions[::seq_stride])

                ped_ids = [[pid]] * len(boxes)
                pids_seq.append(ped_ids[::seq_stride])

                if 'b' not in pid:
                    intent = [[0]] * len(boxes)
                    acts = [[0]] * len(boxes)
                else:
                    crossing_val = pid_annots[pid]['attributes'].get('crossing', -1)
                    if crossing_val == -1:
                        intent = [[0]] * len(boxes)
                    else:
                        intent = [[1]] * len(boxes)
                    acts = [[int(crossing_val > 0)]] * len(boxes)

                intent_seq.append(intent[::seq_stride])
                activities.append(acts[::seq_stride])
                vehicle_seq.append([[vid_annots[i]] for i in frame_ids][::seq_stride])

                # --- ADDED for acc ---
                all_acc = all_acc[::seq_stride] if all_acc else []
                acc_seq.append(all_acc)

        print('Split: %s' % image_set)
        print('Number of pedestrians: %d ' % num_pedestrians)
        print('Total number of samples: %d ' % len(image_seq))

        return {
            'image': image_seq,
            'pid': pids_seq,
            'bbox': box_seq,
            'center': center_seq,
            'occlusion': occ_seq,
            'vehicle_act': vehicle_seq,
            'intent': intent_seq,
            'activities': activities,
            'image_dimension': (img_width, img_height),
            # --- ADDED for acc ---
            'acc': acc_seq
        }

    def _get_intention(self, image_set, annot_database, **params):
        print('---------------------------------------------------------')
        print("Generating intention data")
        num_pedestrians = 0
        seq_stride = params['fstride']
        sq_ratio = params['squarify_ratio']
        height_rng = params['height_rng']

        image_seq, pids_seq = [], []
        box_seq, center_seq, occ_seq = [], [], []
        intent_seq = []

        # --- ADDED for acc ---
        acc_seq = []

        videos_ids, _pids = self._get_data_ids(image_set, params)
        for vid in sorted(videos_ids):
            img_width = annot_database[vid]['width']
            pid_annots = annot_database[vid]['ped_annotations']
            for pid in sorted(pid_annots):
                if params['data_split_type'] != 'default' and pid not in _pids:
                    continue
                if 'p' in pid:
                    continue
                if params['sample_type'] == 'beh' and 'b' not in pid:
                    continue

                num_pedestrians += 1
                frame_ids = pid_annots[pid]['frames']

                if 'b' in pid:
                    event_frame = pid_annots[pid]['attributes'].get('decision_point', -1)
                else:
                    event_frame = -1

                if event_frame == -1:
                    end_idx = -3
                else:
                    end_idx = frame_ids.index(event_frame)

                boxes = pid_annots[pid]['bbox'][:end_idx + 1]
                frame_ids = frame_ids[:end_idx + 1]
                images = [self._get_image_path(vid, f) for f in frame_ids]
                occlusions = pid_annots[pid]['occlusion'][:end_idx + 1]

                # --- ADDED for acc ---
                if 'acc' in pid_annots[pid]:
                    all_acc = pid_annots[pid]['acc'][:end_idx + 1]
                else:
                    all_acc = []

                if height_rng[0] > 0 or height_rng[1] < float('inf'):
                    images, boxes, frame_ids, occlusions = self._height_check(height_rng, frame_ids, boxes, images, occlusions)
                    if all_acc:
                        all_acc = [all_acc[i] for i in range(len(all_acc)) if i < len(images)]

                if len(boxes) / seq_stride < params['min_track_size']:
                    continue

                if sq_ratio:
                    boxes = [self._squarify(b, sq_ratio, img_width) for b in boxes]

                center_seq.append([self._get_center(b) for b in boxes][::seq_stride])
                image_seq.append(images[::seq_stride])
                box_seq.append(boxes[::seq_stride])
                occ_seq.append(occlusions[::seq_stride])

                ped_ids = [[pid]] * len(boxes)
                pids_seq.append(ped_ids[::seq_stride])

                if 'b' not in pid:
                    intent = [[0]] * len(boxes)
                else:
                    crossing_val = pid_annots[pid]['attributes'].get('crossing', -1)
                    if crossing_val == -1:
                        intent = [[0]] * len(boxes)
                    else:
                        intent = [[1]] * len(boxes)

                intent_seq.append(intent[::seq_stride])

                # --- ADDED for acc ---
                all_acc = all_acc[::seq_stride] if all_acc else []
                acc_seq.append(all_acc)

        print('Split: %s' % image_set)
        print('Number of pedestrians: %d ' % num_pedestrians)
        print('Total number of samples: %d ' % len(image_seq))

        return {
            'image': image_seq,
            'pid': pids_seq,
            'bbox': box_seq,
            'center': center_seq,
            'occlusion': occ_seq,
            'intent': intent_seq,
            # --- ADDED for acc ---
            'acc': acc_seq
        }



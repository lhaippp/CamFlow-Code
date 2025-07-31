import os
import cv2
import h5py
import torch
import random
import pickle
import imageio
import logging

import numpy as np
import torch.utils.data

from functools import cached_property
from torch.utils.data import DataLoader, Dataset, RandomSampler, ConcatDataset
from utils_operations.flow_and_mapping_operations import convert_mapping_to_flow, from_homography_to_pixel_wise_mapping


def worker_init_fn(worker_id):
    rand_seed = random.randint(0, 2**32 - 1)
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)


def homo_scale(h0, w0, H, h1, w1):
    M_0 = np.array([[w0 / 2.0, 0., w0 / 2.0], [0., h0 / 2.0, h0 / 2.0],
                    [0., 0., 1.]])
    M_0_inv = np.linalg.inv(M_0)
    H_0_norm = np.matmul(np.matmul(M_0_inv, H), M_0)

    M_1 = np.array([[w1 / 2.0, 0., w1 / 2.0], [0., h1 / 2.0, h1 / 2.0],
                    [0., 0., 1.]])
    M_1_inv = np.linalg.inv(M_1)
    H_1 = np.matmul(np.matmul(M_1, H_0_norm), M_1_inv)
    return H_1


def homo_convert_to_flow(H, size=(360, 640)):

    mapping_from_homography_x, mapping_from_homography_y = from_homography_to_pixel_wise_mapping(
        size, H)

    mapping_from_homography_numpy = np.dstack(
        (mapping_from_homography_x, mapping_from_homography_y))
    flow = convert_mapping_to_flow(
        torch.from_numpy(mapping_from_homography_numpy).unsqueeze(0).permute(
            0, 3, 1, 2))
    return flow.detach().cpu().requires_grad_(False)


class UnHomoTrainData(Dataset):

    def __init__(self, params, phase='train'):
        assert phase in ['train', 'val', 'test']
        # 参数预设

        self.mean_I = np.array([118.93, 113.97, 102.60]).reshape(1, 1, 3)
        self.std_I = np.array([69.85, 68.81, 72.45]).reshape(1, 1, 3)

        self.params = params
        self.crop_size = self.params.crop_size
        self.ori_h, self.ori_w = self.params.ori_size[0], self.params.ori_size[
            1]
        self.rho = self.params.rho

        self.trainset_pth = '/mnt/exp/Contant-Aware-DeepH-Data/Data/Train'
        self.pseudo_labels = np.load(
            '/mnt/exp/Contant-Aware-DeepH-Data/Data/Train/BasesHomo_small.npy',
            allow_pickle=True).item()
        self.im1_im2_names = list(self.pseudo_labels.keys())

        self.seed = 0
        random.seed(self.seed)

        print(f"UnHomoTrainData load {len(self.im1_im2_names)} data from oss!")
        self.cnt = 0

    def __len__(self):
        return len(self.im1_im2_names)

    def get_data(self, idx):
        # img loading
        im1_im2_name = self.im1_im2_names[idx]

        dir_name = im1_im2_name.split('_')[0]
        im1_name = "_".join(im1_im2_name.split('_')[:2]) + '.png'
        im2_name = "_".join(im1_im2_name.split('_')[2:]) + '.png'

        img1 = cv2.imread(os.path.join(self.trainset_pth, dir_name,
                                       im1_name)).astype(np.float32)
        img2 = cv2.imread(os.path.join(self.trainset_pth, dir_name,
                                       im2_name)).astype(np.float32)

        pseudo_label = self.pseudo_labels[im1_im2_name]
        # pseudo_label包含list: [homo_b, homo_f]
        # homo_f is from img1 -> img2, homo_b is from img2 -> img1
        homo_b, homo_f = pseudo_label[0], pseudo_label[1]

        return img1, img2, homo_f, im1_im2_name

    def __getitem__(self, idx):
        img1, img2, homo_gt, im1_im2_name = self.get_data(idx)

        imgs_rgb_full = torch.cat((torch.Tensor(img1), torch.Tensor(img2)),
                                  dim=-1).permute(2, 0, 1).float()

        # unit test for async nori
        # self.cnt += 1
        # img1_warp = cv2.warpPerspective(img1, homo_gt, (256, 256))
        # imageio.mimsave(
        #     f"unit_test/test_async_nori_{self.cnt}.gif",
        #     [
        #         np.concatenate((img1, img1_warp), 1)[:, :, ::-1],
        #         np.concatenate((img2, img2), 1)[:, :, ::-1],
        #     ],
        #     'GIF',
        #     duration=0.5,
        # )
        # if self.cnt > 10:
        #     raise Exception
        if False:
            self.cnt += 1
            img1_warp = cv2.warpPerspective(img1, homo_gt,
                                            img1.shape[:2][::-1])

            def xxx(x):
                return (x * 255).clip(0, 255).astype(np.uint8)

            imageio.mimsave(f"unit_test/test_async_nori_{self.cnt}.gif",
                            [xxx(img2[..., ::-1]),
                             xxx(img1_warp[..., ::-1])],
                            'GIF',
                            duration=0.5,
                            loop=0)
            if self.cnt > 10:
                raise Exception

        h, w, c = img1.shape

        homo_gt = homo_scale(h, w, homo_gt, self.ori_h, self.ori_w)
        homo_gt_inv = np.linalg.inv(homo_gt)

        img1, img2 = cv2.resize(img1, (self.ori_w, self.ori_h)), cv2.resize(
            img2, (self.ori_w, self.ori_h))

        img1, img2, img1_patch, img2_patch, flow_gt_b, flow_gt_f, flow_gt_b_patch, flow_gt_f_patch, start = self.data_aug(
            img1, img2, homo_gt, homo_gt_inv)

        imgs_gray_full = torch.cat((img1, img2), dim=2).permute(2, 0,
                                                                1).float()
        imgs_gray_patch = torch.cat((img1_patch, img2_patch),
                                    dim=2).permute(2, 0, 1).float()
        flow_gt_full = torch.cat((flow_gt_b, flow_gt_f), dim=1).squeeze(0)
        flow_gt_patch = torch.cat((flow_gt_b_patch, flow_gt_f_patch),
                                  dim=1).squeeze(0)
        start = torch.Tensor(start).reshape(2, 1, 1).float()
        # output dict
        data_dict = {
            "imgs_gray_full": imgs_gray_full,
            "imgs_gray_patch": imgs_gray_patch,
            "flow_gt_full": flow_gt_full,
            "flow_gt_patch": flow_gt_patch,
            "start": start,
            "ganhomo_mask": torch.ones_like(imgs_gray_patch),
            "imgs_rgb_full": imgs_rgb_full,
            'im1_im2_name': im1_im2_name
        }

        return data_dict

    def data_aug(self,
                 img1,
                 img2,
                 homo_gt,
                 homo_gt_inv,
                 start=None,
                 normalize=True,
                 gray=True):

        def random_crop_tt(img1, img2, homo_gt, homo_gt_inv, start):
            height, width = img1.shape[:2]
            patch_size_h, patch_size_w = self.crop_size

            if start is None:
                x = random.randint(self.rho, width - self.rho - patch_size_w)
                y = random.randint(self.rho, height - self.rho - patch_size_h)
                start = [x, y]
            else:
                x, y = start
            img1_patch = img1[y:y + patch_size_h, x:x + patch_size_w, :]
            img2_patch = img2[y:y + patch_size_h, x:x + patch_size_w, :]

            flow_gt_b = homo_convert_to_flow(homo_gt_inv, (height, width))
            flow_gt_f = homo_convert_to_flow(homo_gt, (height, width))
            flow_gt_b_patch = flow_gt_b[:, :, y:y + patch_size_h,
                                        x:x + patch_size_w]
            flow_gt_f_patch = flow_gt_f[:, :, y:y + patch_size_h,
                                        x:x + patch_size_w]
            return img1, img2, img1_patch, img2_patch, flow_gt_b, flow_gt_f, flow_gt_b_patch, flow_gt_f_patch, start

        if normalize:
            img1 = (img1 - self.mean_I) / self.std_I
            img2 = (img2 - self.mean_I) / self.std_I

        if gray:
            img1 = np.mean(img1, axis=2, keepdims=True)
            img2 = np.mean(img2, axis=2, keepdims=True)

        img1, img2 = list(map(torch.Tensor, [img1, img2]))

        img1, img2, img1_patch, img2_patch, flow_gt_b, flow_gt_f, flow_gt_b_patch, flow_gt_f_patch, start \
            = random_crop_tt(img1, img2, homo_gt, homo_gt_inv, start)

        return img1, img2, img1_patch, img2_patch, flow_gt_b, flow_gt_f, flow_gt_b_patch, flow_gt_f_patch, start


class UnHomoTrainDataDMHomo(UnHomoTrainData):
    """
    新的dataset class，使用flow和mask数据替代原来的homography数据
    继承自UnHomoTrainData，主要更换pseudo_labels的加载方式
    """

    def __init__(self, params, phase='train'):
        # 先调用父类的初始化，但会覆盖pseudo_labels
        super().__init__(params, phase)

        # 更换pseudo_labels的加载路径
        self.pseudo_labels_dir = '/mnt/exp/CamFlowRebuttal/experiments/CAHomo/dmhomo_gt/'

        # 重新构建im1_im2_names列表，基于实际存在的.npy文件
        self.im1_im2_names = []
        for name in os.listdir(self.pseudo_labels_dir):
            if name.endswith('.npy'):
                # 移除.npy后缀
                self.im1_im2_names.append(name[:-4])

        print(
            f"UnHomoTrainDataDMHomo load {len(self.im1_im2_names)} data from {self.pseudo_labels_dir}!"
        )

        # 初始化计数器，用于测试
        self.cnt = 0

    def get_data(self, idx):
        # img loading
        im1_im2_name = self.im1_im2_names[idx]

        dir_name = im1_im2_name.split('_')[0]
        im1_name = "_".join(im1_im2_name.split('_')[:2]) + '.png'
        im2_name = "_".join(im1_im2_name.split('_')[2:]) + '.png'

        img1 = cv2.imread(os.path.join(self.trainset_pth, dir_name,
                                       im1_name)).astype(np.float32)
        img2 = cv2.imread(os.path.join(self.trainset_pth, dir_name,
                                       im2_name)).astype(np.float32)

        # 加载新的pseudo_label数据（包含flow和mask）
        save_path = os.path.join(self.pseudo_labels_dir, f"{im1_im2_name}.npy")
        pseudo_label_data = np.load(save_path, allow_pickle=True).item()

        # 从npy文件中提取flow数据
        # npy里面有4个东西: flow_f, flow_b, mask_f, mask_b
        flow_f = pseudo_label_data['flow_f']  # forward flow
        flow_b = pseudo_label_data['flow_b']  # backward flow

        # Change flow shape from [2, h, w] to [h, w, 2]
        flow_f = np.transpose(flow_f, (1, 2, 0))
        flow_b = np.transpose(flow_b, (1, 2, 0))

        return img1, img2, flow_f, flow_b, im1_im2_name

    def resize_flow(self, flow, target_h, target_w):
        """
        Resize flow to [target_h, target_w, 2] if needed (no scaling applied).
        flow: np.ndarray of shape [h, w, 2]
        Returns: np.ndarray of shape [target_h, target_w, 2]
        """
        assert flow.ndim == 3 and flow.shape[
            2] == 2, f"Flow shape must be [h, w, 2], got {flow.shape}"
        h, w, _ = flow.shape
        if (h, w) == (target_h, target_w):
            return flow

        # 直接resize整个flow，保持[h,w,2]格式，不需要缩放
        flow_resized = cv2.resize(flow, (target_w, target_h),
                                  interpolation=cv2.INTER_LINEAR)

        return flow_resized

    def __getitem__(self, idx):
        img1, img2, flow_f, flow_b, im1_im2_name = self.get_data(idx)

        h_img, w_img = img1.shape[:2]
        # Ensure flow shape is [h, w, 2] and matches image spatial size
        flow_f = self.resize_flow(flow_f, h_img, w_img)
        flow_b = self.resize_flow(flow_b, h_img, w_img)

        imgs_rgb_full = torch.cat((torch.Tensor(img1), torch.Tensor(img2)),
                                  dim=-1).permute(2, 0, 1).float()

        if False:  # 可以设置为True来启用测试
            self.cnt += 1
            h, w = img1.shape[:2]

            # 创建网格坐标并使用flow进行remap
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = x.astype(np.float32) + flow_f[:, :, 0]
            map_y = y.astype(np.float32) + flow_f[:, :, 1]
            img2_warped = cv2.remap(img2, map_x, map_y, cv2.INTER_LINEAR)

            def xxx(x):
                return (x).clip(0, 255).astype(np.uint8)

            # 保存gif验证flow对应关系
            import imageio
            import os
            os.makedirs("unit_test", exist_ok=True)
            imageio.mimsave(
                f"unit_test/test_dmhomo_flow_{self.cnt}.gif",
                [xxx(img1[..., ::-1]),
                 xxx(img2_warped[..., ::-1])],
                duration=0.5,
                loop=0)
            if self.cnt > 10:
                raise Exception

        h, w, c = img1.shape

        # 调整图像尺寸
        # Resize images to (self.ori_w, self.ori_h)
        h0, w0 = img1.shape[:2]
        img1 = cv2.resize(img1, (self.ori_w, self.ori_h))
        img2 = cv2.resize(img2, (self.ori_w, self.ori_h))
        # If image is resized, resize and scale the flow accordingly
        if (h0, w0) != (self.ori_h, self.ori_w):
            scale_x = self.ori_w / w0
            scale_y = self.ori_h / h0

            def resize_and_scale_flow(flow):
                flow_resized = cv2.resize(flow, (self.ori_w, self.ori_h),
                                          interpolation=cv2.INTER_LINEAR)
                flow_resized[..., 0] *= scale_x
                flow_resized[..., 1] *= scale_y
                return flow_resized

            flow_f = resize_and_scale_flow(flow_f)
            flow_b = resize_and_scale_flow(flow_b)

        # 进行数据增强，直接使用加载的flow数据
        img1, img2, img1_patch, img2_patch, flow_gt_b, flow_gt_f, flow_gt_b_patch, flow_gt_f_patch, start = self.data_aug_with_precomputed_flow(
            img1, img2, flow_f, flow_b)

        imgs_gray_full = torch.cat((img1, img2), dim=2).permute(2, 0,
                                                                1).float()
        imgs_gray_patch = torch.cat((img1_patch, img2_patch),
                                    dim=2).permute(2, 0, 1).float()
        flow_gt_full = torch.cat((flow_gt_b, flow_gt_f), dim=1).squeeze(0)
        flow_gt_patch = torch.cat((flow_gt_b_patch, flow_gt_f_patch),
                                  dim=1).squeeze(0)
        start = torch.Tensor(start).reshape(2, 1, 1).float()

        # output dict
        data_dict = {
            "imgs_gray_full": imgs_gray_full,
            "imgs_gray_patch": imgs_gray_patch,
            "flow_gt_full": flow_gt_full,
            "flow_gt_patch": flow_gt_patch,
            "start": start,
            "ganhomo_mask": torch.ones_like(imgs_gray_patch),
            "imgs_rgb_full": imgs_rgb_full,
            'im1_im2_name': im1_im2_name
        }

        return data_dict

    def data_aug_with_precomputed_flow(self,
                                       img1,
                                       img2,
                                       flow_f,
                                       flow_b,
                                       start=None,
                                       normalize=True,
                                       gray=True):
        """
        使用预先计算好的flow数据进行数据增强
        """

        def random_crop_with_precomputed_flow(img1, img2, flow_f, flow_b,
                                              start):
            height, width = img1.shape[:2]
            patch_size_h, patch_size_w = self.crop_size

            if start is None:
                x = random.randint(self.rho, width - self.rho - patch_size_w)
                y = random.randint(self.rho, height - self.rho - patch_size_h)
                start = [x, y]
            else:
                x, y = start

            img1_patch = img1[y:y + patch_size_h, x:x + patch_size_w, :]
            img2_patch = img2[y:y + patch_size_h, x:x + patch_size_w, :]

            # 调整flow尺寸并裁剪
            flow_f_resized = cv2.resize(flow_f, (width, height))
            flow_b_resized = cv2.resize(flow_b, (width, height))

            # 转换为tensor并调整维度
            flow_gt_f = torch.from_numpy(flow_f_resized).permute(
                2, 0, 1).float().unsqueeze(0)
            flow_gt_b = torch.from_numpy(flow_b_resized).permute(
                2, 0, 1).float().unsqueeze(0)

            # 裁剪flow
            flow_gt_f_patch = flow_gt_f[:, :, y:y + patch_size_h,
                                        x:x + patch_size_w]
            flow_gt_b_patch = flow_gt_b[:, :, y:y + patch_size_h,
                                        x:x + patch_size_w]

            return img1, img2, img1_patch, img2_patch, flow_gt_b, flow_gt_f, flow_gt_b_patch, flow_gt_f_patch, start

        if normalize:
            img1 = (img1 - self.mean_I) / self.std_I
            img2 = (img2 - self.mean_I) / self.std_I

        if gray:
            img1 = np.mean(img1, axis=2, keepdims=True)
            img2 = np.mean(img2, axis=2, keepdims=True)

        img1, img2 = list(map(torch.Tensor, [img1, img2]))

        img1, img2, img1_patch, img2_patch, flow_gt_b, flow_gt_f, flow_gt_b_patch, flow_gt_f_patch, start \
            = random_crop_with_precomputed_flow(img1, img2, flow_f, flow_b, start)

        return img1, img2, img1_patch, img2_patch, flow_gt_b, flow_gt_f, flow_gt_b_patch, flow_gt_f_patch, start


class UnHomoTrainDataSup(UnHomoTrainData):

    def __init__(self, params, phase='train'):
        super().__init__(params, phase)

        self.nori_pkl = pickle.load(
            open(
                os.path.join(self.data_dir,
                             f'{self.params.nori_dataset_name_sup}.pickle'),
                'rb'))
        self.nori_pkl_value_list = list(self.nori_pkl.values())

        print(
            f"UnHomoTrainDataSup load {len(self.nori_pkl_value_list)} data from oss!"
        )


class GanHomoTrainData(Dataset):

    def __init__(
        self,
        params,
        phase,
        total_data_slice_idx=1,
        data_slice_idx=0,
    ):
        assert phase in ['em2']
        # 参数预设
        self.homo = np.load(
            '/data/denoising-diffusion-pytorch/work/20230227.9i6o.GanHomo.GanMask.Dilate.Class.AdpL/dataset/HomoGAN_sup.npy',
            allow_pickle=True).item()

        self.nori_info = np.load(
            '/data/denoising-diffusion-pytorch/work/20221230.cond.rgbHomoFlow.9i6o/dataset/train.npy',
            allow_pickle=True).item()
        self.nori_ids, self.nori_length, self.plk_dict = self.nori_info[
            "nori_ids"], self.nori_info["nori_length"], self.nori_info[
                "plk_dict"]

        self.slice_nori_length = (self.nori_length // total_data_slice_idx)
        print(f"self.nori_length {self.slice_nori_length}/{self.nori_length}")
        self.nori_ids = self.nori_ids[data_slice_idx *
                                      self.slice_nori_length:(data_slice_idx +
                                                              1) *
                                      self.slice_nori_length]

        from torchvision import transforms as T
        self.transform = T.Compose([
            T.ToTensor(),
            # T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            # T.CenterCrop(image_size),
        ])

        self.mean_I = np.array([118.93, 113.97, 102.60]).reshape(1, 1, 3)
        self.std_I = np.array([69.85, 68.81, 72.45]).reshape(1, 1, 3)

        self.crop_size = params.crop_size

    @staticmethod
    def bytes2np(data, size, dtype=np.float32):
        data = np.fromstring(data, dtype)
        data = data.reshape(size)
        return data

    def __len__(self):
        print(f"The length of UnHomoTrainData is {self.slice_nori_length}")
        return self.slice_nori_length

    def __getitem__(self, idx):
        # img loading
        images_bytes = fetcher.get(self.nori_ids[idx])
        buf = self.plk_dict[self.nori_ids[idx]]

        images_dtype, images_size = buf['_images_dtype'], buf['_images_size']
        masks_id, masks_dtype, masks_size, file_name = \
            buf['_id_masks'], buf['_masks_dtype'], buf['_masks_size'], buf['idx']
        masks_bytes = fetcher.get(masks_id)

        images = self.bytes2np(images_bytes,
                               size=images_size,
                               dtype=images_dtype).astype(np.float32)
        masks = self.bytes2np(masks_bytes, size=masks_size, dtype=masks_dtype)

        # uint8 [0, 256]
        img1, img2 = images[:, :, :3], images[:, :, 3:]

        # homoGAN mask
        mask = masks[:, :, 1:2].astype(np.float32)
        mask = torch.Tensor(mask)

        img1_rgb, img2_rgb = img1, img2

        imgs_rgb_full = torch.cat(
            (torch.Tensor(img1_rgb), torch.Tensor(img2_rgb)), dim=-1).permute(
                2, 0, 1).float() / 255.

        imgs_full = torch.cat((torch.Tensor(img1), torch.Tensor(img2)),
                              dim=-1).permute(2, 0, 1).float()

        img1 = (img1 - self.mean_I) / self.std_I
        img2 = (img2 - self.mean_I) / self.std_I

        img1 = np.mean(img1, axis=2, keepdims=True)
        img2 = np.mean(img2, axis=2, keepdims=True)

        img1_rs = cv2.resize(img1, (self.crop_size[1], self.crop_size[0]))
        img2_rs = cv2.resize(img2, (self.crop_size[1], self.crop_size[0]))

        img1, img2, img1_rs, img2_rs = list(
            map(torch.Tensor, [img1, img2, img1_rs, img2_rs]))

        imgs_gray_full = torch.cat((img1, img2), dim=-1).permute(2, 0,
                                                                 1).float()
        imgs_gray_patch = torch.cat(
            (img1_rs.unsqueeze(0), img2_rs.unsqueeze(0)), dim=0).float()

        data_dict = {
            "imgs_gray_full": imgs_gray_full,
            "imgs_rgb_full": imgs_rgb_full,
            "imgs_full": imgs_full,
            "imgs_gray_patch": imgs_gray_patch,
            "ganhomo_mask": mask,
        }
        return data_dict


class DGMTrainData(Dataset):

    def __init__(self, db_path: str, crop_shape: tuple[int, int],
                 ori_shape: tuple[int, int], rho: int):
        self.mean_I = np.array([118.93, 113.97, 102.60]).reshape(1, 1, 3)
        self.std_I = np.array([69.85, 68.81, 72.45]).reshape(1, 1, 3)

        self.db_path = db_path
        self.crop_size = crop_shape
        self.ori_h, self.ori_w = ori_shape[0], ori_shape[1]
        self.rho = rho

    def __len__(self):
        print(f"The length of DGMTrainData is {len(self.db)}")
        return len(self.db)

    @cached_property
    def db(self):
        return h5py.File(self.db_path)

    def __getitem__(self, idx):
        _buf = self.db[f"{idx:08d}"]

        homo_f = _buf["homo"][:]
        homo_gt = homo_f

        img1 = cv2.imdecode(np.array(_buf["img1"][:]), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.array(_buf["img2"][:]), cv2.IMREAD_COLOR)
        h, w, _ = img1.shape

        if h != self.ori_h or w != self.ori_w:
            homo_gt = homo_scale(h, w, homo_gt, self.ori_h, self.ori_w)
            homo_gt_inv = np.linalg.inv(homo_gt)

            img1, img2 = cv2.resize(img1,
                                    (self.ori_w, self.ori_h)), cv2.resize(
                                        img2, (self.ori_w, self.ori_h))

        imgs_rgb_full = torch.cat((torch.Tensor(img1), torch.Tensor(img2)),
                                  dim=-1).permute(2, 0, 1).float() / 255.0

        img1, img2, img1_patch, img2_patch, flow_gt_b, flow_gt_f, flow_gt_b_patch, flow_gt_f_patch, start = (
            self.data_aug(img1, img2, homo_gt, homo_gt_inv))

        imgs_gray_full = torch.cat((img1, img2), dim=2).permute(2, 0,
                                                                1).float()
        imgs_gray_patch = torch.cat((img1_patch, img2_patch),
                                    dim=2).permute(2, 0, 1).float()
        flow_gt_full = torch.cat((flow_gt_b, flow_gt_f), dim=1).squeeze(0)
        flow_gt_patch = torch.cat((flow_gt_b_patch, flow_gt_f_patch),
                                  dim=1).squeeze(0)
        start = torch.Tensor(start).reshape(2, 1, 1).float()

        # output dict
        data_dict = {
            "imgs_gray_full": imgs_gray_full,
            "imgs_gray_patch": imgs_gray_patch,
            "flow_gt_full": flow_gt_full,
            "flow_gt_patch": flow_gt_patch,
            "start": start,
            "imgs_rgb_full": imgs_rgb_full,
        }
        return data_dict

    def data_aug(self,
                 img1,
                 img2,
                 homo_gt,
                 homo_gt_inv,
                 start=None,
                 normalize=True,
                 gray=True):

        def random_crop_tt(img1, img2, homo_gt, homo_gt_inv, start):
            height, width = img1.shape[:2]
            patch_size_h, patch_size_w = self.crop_size

            if start is None:
                x = random.randint(self.rho, width - self.rho - patch_size_w)
                y = random.randint(self.rho, height - self.rho - patch_size_h)
                start = [x, y]
            else:
                x, y = start
            img1_patch = img1[y:y + patch_size_h, x:x + patch_size_w, :]
            img2_patch = img2[y:y + patch_size_h, x:x + patch_size_w, :]

            flow_gt_b = homo_convert_to_flow(homo_gt_inv, (height, width))
            flow_gt_f = homo_convert_to_flow(homo_gt, (height, width))
            flow_gt_b_patch = flow_gt_b[:, :, y:y + patch_size_h,
                                        x:x + patch_size_w]
            flow_gt_f_patch = flow_gt_f[:, :, y:y + patch_size_h,
                                        x:x + patch_size_w]
            return img1, img2, img1_patch, img2_patch, flow_gt_b, flow_gt_f, flow_gt_b_patch, flow_gt_f_patch, start

        if normalize:
            img1 = (img1 - self.mean_I) / self.std_I
            img2 = (img2 - self.mean_I) / self.std_I

        if gray:
            img1 = np.mean(img1, axis=2, keepdims=True)
            img2 = np.mean(img2, axis=2, keepdims=True)

        img1, img2 = list(map(torch.Tensor, [img1, img2]))

        img1, img2, img1_patch, img2_patch, flow_gt_b, flow_gt_f, flow_gt_b_patch, flow_gt_f_patch, start = (
            random_crop_tt(img1, img2, homo_gt, homo_gt_inv, start))

        return img1, img2, img1_patch, img2_patch, flow_gt_b, flow_gt_f, flow_gt_b_patch, flow_gt_f_patch, start


class HomoTestData(Dataset):

    def __init__(self, params, phase):
        assert phase in ["test", "val"]

        self.crop_size = params.crop_size
        self.patch_size_h, self.patch_size_w = params.crop_size
        self.generate_size = params.generate_size

        self.data_list = os.path.join(params.test_data_dir, "test.txt")
        self.npy_path = os.path.join(params.test_data_dir, "pt")
        self.image_path = os.path.join(params.test_data_dir, "img")
        self.data_infor = open(self.data_list, 'r').readlines()

        self.mean_I = np.array([118.93, 113.97, 102.60]).reshape(1, 1, 3)
        self.std_I = np.array([69.85, 68.81, 72.45]).reshape(1, 1, 3)

        print(f"HomoTestData length {len(self.data_infor)}")

    def __len__(self):
        return len(self.data_infor)

    def __getitem__(self, idx):
        img_names = self.data_infor[idx].replace('\n', '')
        video_names = img_names.split('/')[0]
        img_names = img_names.split(' ')
        pt_names = img_names[0].split('/')[-1] + '_' + img_names[1].split(
            '/')[-1] + '.npy'
        save_name = img_names[0].split('.')[0].split(
            '/')[1] + '_' + img_names[1].split('.')[0].split('/')[1]

        img1 = cv2.imread(os.path.join(self.image_path, img_names[0]))
        img2 = cv2.imread(os.path.join(self.image_path, img_names[1]))

        img1_rgb, img2_rgb = img1, img2
        imgs_rgb_full = torch.cat(
            (torch.Tensor(img1_rgb), torch.Tensor(img2_rgb)), dim=-1).permute(
                2, 0, 1).float() / 255.

        imgs_full = torch.cat((torch.Tensor(img1), torch.Tensor(img2)),
                              dim=-1).permute(2, 0, 1).float()
        ori_h, ori_w, _ = img1.shape

        # 模拟diffusion生成数据的256尺寸
        # img1 = cv2.resize(img1, (self.generate_size, self.generate_size))
        # img2 = cv2.resize(img2, (self.generate_size, self.generate_size))
        # print(f'img1 shape {img1.shape}')

        pt_set = np.load(os.path.join(self.npy_path, pt_names),
                         allow_pickle=True).item()
        pt_set = torch.Tensor(pt_set["matche_pts"][:6])

        img1 = (img1 - self.mean_I) / self.std_I
        img2 = (img2 - self.mean_I) / self.std_I

        img1 = np.mean(img1, axis=2, keepdims=True)
        img2 = np.mean(img2, axis=2, keepdims=True)

        img1_rs = cv2.resize(img1, (self.crop_size[1], self.crop_size[0]))
        img2_rs = cv2.resize(img2, (self.crop_size[1], self.crop_size[0]))

        # 模拟diffusion生成数据从 256 -> 原尺寸
        # img1 = cv2.resize(img1, (ori_w, ori_h)).unsqueeze(0)
        # img2 = cv2.resize(img2, (ori_w, ori_h)).unsqueeze(0)

        img1, img2, img1_rs, img2_rs = list(
            map(torch.Tensor, [img1, img2, img1_rs, img2_rs]))

        imgs_gray_full = torch.cat((img1, img2), dim=-1).permute(2, 0,
                                                                 1).float()
        imgs_gray_patch = torch.cat(
            (img1_rs.unsqueeze(0), img2_rs.unsqueeze(0)), dim=0).float()

        ori_size = torch.Tensor([ori_w, ori_h]).float()
        Ph, Pw = img1_rs.size()

        pts = torch.Tensor([[0, 0], [Pw - 1, 0], [0, Ph - 1],
                            [Pw - 1, Ph - 1]]).float()
        start = torch.Tensor([0, 0]).reshape(2, 1, 1).float()

        data_dict = {
            "imgs_gray_full": imgs_gray_full,
            "imgs_rgb_full": imgs_rgb_full,
            "imgs_full": imgs_full,
            "imgs_gray_patch": imgs_gray_patch,
            "ori_size": ori_size,
            "pt_set": pt_set,
            'pt_names': pt_names,
            "video_names": video_names,
            "pts": pts,
            "start": start,
            "save_name": save_name,
            "ganhomo_mask": torch.ones_like(imgs_full),
        }
        return data_dict


class GhofBMK(Dataset):

    def __init__(self, params, phase):
        assert phase in ["test", "val"]

        # Use the merged file instead of separate data and mask files
        self.path = './GHOF-Cam.npy'
        self.data = np.load(self.path, allow_pickle=True)

        # No need to load mask separately since it's now included in the data
        # self.mask_pth = '/mnt/exp/CameraFlowMetric/GHOF_Clean_20230705_mask_merged.npy'
        # self.mask = np.load(self.mask_pth, allow_pickle=True)

        # No need for assertion since mask is now part of data
        # assert len(self.mask) == len(
        #     self.data
        # ), f"mask length {len(self.mask)} != data length {len(self.data)}"
        print(f"GhofBMK length {len(self.data)}")

        self.crop_size = params.crop_size
        self.patch_size_h, self.patch_size_w = params.crop_size
        self.generate_size = params.generate_size

        self.mean_I = np.array([118.93, 113.97, 102.60]).reshape(1, 1, 3)
        self.std_I = np.array([69.85, 68.81, 72.45]).reshape(1, 1, 3)

        self.original_length = len(self.data)  # 保存原始样本数（254）

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img1, img2 = self.data[idx]["img1"], self.data[idx]["img2"]
        category = self.data[idx]["split"]
        mask = self.data[idx]["mask"]  # Now get mask from the same data structure

        gt_flow, homo_field = self.data[idx]["gt_flow"], self.data[idx][
            "homo_field"]

        gt_flow = torch.from_numpy(gt_flow).permute(2, 0, 1)
        homo_field = torch.from_numpy(homo_field).permute(2, 0, 1)
        mask = torch.from_numpy(mask[None])

        img1_rgb, img2_rgb = img1, img2
        imgs_rgb_full = torch.cat(
            (torch.Tensor(img1_rgb), torch.Tensor(img2_rgb)), dim=-1).permute(
                2, 0, 1).float() / 255.

        imgs_full = torch.cat((torch.Tensor(img1), torch.Tensor(img2)),
                              dim=-1).permute(2, 0, 1).float()
        ori_h, ori_w, _ = img1.shape

        img1 = (img1 - self.mean_I) / self.std_I
        img2 = (img2 - self.mean_I) / self.std_I

        img1 = np.mean(img1, axis=2, keepdims=True)
        img2 = np.mean(img2, axis=2, keepdims=True)

        img1_rs = cv2.resize(img1, (self.crop_size[1], self.crop_size[0]))
        img2_rs = cv2.resize(img2, (self.crop_size[1], self.crop_size[0]))

        img1, img2, img1_rs, img2_rs = list(
            map(torch.Tensor, [img1, img2, img1_rs, img2_rs]))

        imgs_gray_full = torch.cat((img1, img2), dim=-1).permute(2, 0,
                                                                 1).float()
        imgs_gray_patch = torch.cat(
            (img1_rs.unsqueeze(0), img2_rs.unsqueeze(0)), dim=0).float()

        ori_size = torch.Tensor([ori_w, ori_h]).float()
        Ph, Pw = img1_rs.size()

        pts = torch.Tensor([[0, 0], [Pw - 1, 0], [0, Ph - 1],
                            [Pw - 1, Ph - 1]]).float()
        start = torch.Tensor([0, 0]).reshape(2, 1, 1).float()

        data_dict = {
            "imgs_gray_full": imgs_gray_full,
            "imgs_rgb_full": imgs_rgb_full,
            "imgs_full": imgs_full,
            "imgs_gray_patch": imgs_gray_patch,
            "ori_size": ori_size,
            "pts": pts,
            "start": start,
            "ganhomo_mask": torch.ones_like(imgs_full),
            "video_names": category,
            "gt_flow": gt_flow,
            "homo_field": homo_field,
            "mask": mask,
            "orig_idx": idx,  # 添加原始索引
            "is_valid": idx < 254  # 标识有效样本
        }
        return data_dict


def collate_fn(batch):
    # batch = list(filter(lambda x: x is not None, batch))
    batch = list(filter(lambda x: x is not None and len(x) > 0, batch))
    if len(batch) == 0:
        return None  # 这里返回 None，让 DataLoader 处理空 batch 情况
    return torch.utils.data.dataloader.default_collate(batch)
    # if len(batch) == 0:
    #     return None


def custom_collate(batch):
    # 手动拼接所有字段
    collated = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([item[key] for item in batch])
        else:  # 处理非张量字段（如 is_valid）
            collated[key] = [item[key] for item in batch]
    return collated


def fetch_dataloader(params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        status_manager: (class) status_manager

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """

    # for psedo labels generating
    # train_ds = GanHomoTrainData(params, phase='em2')
    # train_ds = ConcatDataset([UnHomoTrainData(params, phase='train'), UnHomoTrainDataSup(params, phase='train')])

    # train_ds = UnHomoTrainDataDMHomo(params, phase='train')
    # train_ds = DGMTrainData(params.db_path, params.crop_size, params.ori_size,
    #                         params.rho)

    # train_ds = UnHomoTrainData(params, phase='train')

    if hasattr(params, "trainset"):
        print(f"trainset: {params.trainset}")
        if params.trainset == "dgm":
            train_ds = DGMTrainData(params.db_path, params.crop_size,
                                    params.ori_size, params.rho)
        elif params.trainset == "cahomo_dmhomo":
            train_ds = UnHomoTrainDataDMHomo(params, phase='train')
        else:
            raise ValueError(
                f"Unknown trainset in params, should in [dgm, cahomo]")
    else:
        train_ds = DGMTrainData(params.db_path, params.crop_size,
                                params.ori_size, params.rho)

    val_ds = HomoTestData(params, phase='val')
    test_ds = HomoTestData(params, phase='test')
    test_ds_ghof = GhofBMK(params, phase='test')

    # train_sampler = RandomSampler(train_ds,
    #                               num_samples=params.partial_data_number)

    dataloaders = {}

    # add defalt train data loader
    train_dl = DataLoader(
        train_ds,
        batch_size=params.train_batch_size,
        shuffle=True,
        num_workers=params.num_workers,
        pin_memory=params.cuda,
        drop_last=True,
        # worker_init_fn=worker_init_fn,
        collate_fn=collate_fn,
        # sampler=train_sampler,
    )
    dataloaders["train"] = train_dl

    dataloaders["ghof"] = DataLoader(
        test_ds_ghof,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=params.cuda,
    )

    # chosse val or test data loader for evaluate
    for split in ["val", "test"]:
        if split in params.eval_type:
            if split == "val":
                dl = DataLoader(
                    val_ds,
                    batch_size=params.eval_batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=params.cuda,
                )
            elif split == "test":
                dl = DataLoader(
                    test_ds,
                    batch_size=params.eval_batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=params.cuda,
                )
            else:
                raise ValueError(
                    "Unknown eval_type in params, should in [val, test]")
            dataloaders[split] = dl
        else:
            dataloaders[split] = None

    return dataloaders

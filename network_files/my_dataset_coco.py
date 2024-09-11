import math
import os
import json

import torch
from PIL import Image
import torch.utils.data as data
from pycocotools.coco import COCO
from train_utils import coco_remove_images_without_annotations, convert_coco_poly_mask


class CocoDetection(data.Dataset):
    """`MS Coco Detection <https://cocodataset.org/>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        dataset (string): train or val.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, dataset="train", transforms=None):
        super(CocoDetection, self).__init__()
        if dataset == "train":
            filename = "training_annotations.json"
        elif dataset == "val":
            filename = "validation_annotations.json"
        else:
            raise ValueError('dataset must be "train" or "val"')

        self.img_root = root

        self.anno_path = os.path.join(root, filename)

        self.mode = dataset
        self.transforms = transforms
        self.coco = COCO(self.anno_path)

        # 获取coco数据索引与类别名称的关系
        # 注意在object80中的索引并不是连续的，虽然只有80个类别，但索引还是按照stuff91来排序的
        data_classes = dict([(v["id"], v["name"]) for k, v in self.coco.cats.items()])
        max_index = max(data_classes.keys())  # 90
        # 将缺失的类别名称设置成N/A
        coco_classes = {}
        for k in range(1, max_index + 1):
            if k in data_classes:
                coco_classes[k] = data_classes[k]
            else:
                coco_classes[k] = "N/A"

        if dataset == "train":
            json_str = json.dumps(coco_classes, indent=4)
            with open("mycls.json", "w") as f:
                f.write(json_str)

        self.coco_classes = coco_classes

        ids = list(sorted(self.coco.imgs.keys()))
        if dataset == "train":
            # 移除没有目标，或者目标面积非常小的数据
            valid_ids = coco_remove_images_without_annotations(self.coco, ids)
            self.ids = valid_ids
        else:
            self.ids = ids

    def parse_targets(self,
                      img_id: int,
                      coco_targets: list,
                      w: int = None,
                      h: int = None):
        assert w > 0
        assert h > 0

        # 只筛选出单个对象的情况
        anno = [obj for obj in coco_targets if obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]

        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # [xmin, ymin, w, h] -> [xmin, ymin, xmax, ymax]
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])

        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_mask(segmentations, h, w)

        # 处理 grasps
        grasps = []
        for obj in anno:
            if "grasps" in obj:
                grasps.extend(obj["grasps"])

        # 检查 grasps 数据格式
        for grasp in grasps:
            if len(grasp) != 5:
                raise ValueError(f"Expected sequence of length 5, but got {len(grasp)}. Grasp: {grasp}")

        # 将 grasps 转换为 Tensor 格式
        grasps = torch.tensor(grasps, dtype=torch.float32).reshape(-1, 5)
        grasps[:, 0] = (grasps[:, 0] - grasps[:, 2] / 2.0).clamp(min=0, max=w)
        grasps[:, 1] = (grasps[:, 1] - grasps[:, 3] / 2.0).clamp(min=0, max=h)
        # 计算 xmax 和 ymax
        grasps[:, 2] = (grasps[:, 0] + grasps[:, 2] / 2.0).clamp(min=0, max=w)
        grasps[:, 3] = (grasps[:, 1] + grasps[:, 3] / 2.0).clamp(min=0, max=h)
        # t 值保持不变，移动到第五列
        # 定义角度区间与标签的映射关系
        angle_to_label = {
            (5, 15): 1,
            (15, 25): 2,
            (25, 35): 3,
            (35, 45): 4,
            (45, 55): 5,
            (55, 65): 6,
            (65, 75): 7,
            (75, 85): 8,
            (85, 90) and (-90, -85): 9,
            (-85, -75): 10,
            (-75, -65): 11,
            (-65, -55): 12,
            (-55, -45): 13,
            (-45, -35): 14,
            (-35, -25): 15,
            (-25, -15): 16,
            (-15, -5): 17,
        }
        # 将 grasps 中的 't' 值转换为对应的标签
        for i in range(grasps.size(0)):
            angle = math.degrees(-grasps[i, 4])
            label = 0  # 默认标签值
            for angles, lab in angle_to_label.items():
                if angles[0] <= angle <= angles[1]:
                    label = lab
                    break
            grasps[i, 4] = label  # 更新标签值到抓取框的第五个元素
        # 不再依赖于 boxes 的筛选结果，直接使用所有的注释数据
        # 使用 .clone().detach() 来构造 Tensor
        classes = torch.tensor(classes.clone().detach(), dtype=torch.int64)
        area = torch.tensor(area.clone().detach())
        iscrowd = torch.tensor(iscrowd.clone().detach())
        masks = torch.tensor(masks.clone().detach())

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = torch.tensor([img_id])
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["grasps"] = grasps[:, :4]  # 添加 grasps 信息
        target["angle_labels"] = grasps[:, 4].long()

        return target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.img_root, path)).convert('RGB')

        w, h = img.size
        target = self.parse_targets(img_id, coco_target, w, h)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def get_height_and_width(self, index):
        coco = self.coco
        img_id = self.ids[index]

        img_info = coco.loadImgs(img_id)[0]
        w = img_info["width"]
        h = img_info["height"]
        return h, w

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


if __name__ == '__main__':
    train = CocoDetection(r"C:\Users\86156\Desktop\grasping_siamese_mask_rcnn-main\data\OCID_grasp", dataset="val")
    print(len(train))
    t = train[0]

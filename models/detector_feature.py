"""
Original Source: https://github.com/rowanz/r2c/blob/master/utils/detector_new.py
"""

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.model_zoo as model_zoo
from torchvision.models import resnet

from transformers.modeling_bert import BertLayerNorm as LayerNorm

from models.pytorch_misc import pad_sequence

USE_IMAGENET_PRETRAINED = True

def _load_resnet(pretrained=True):
    # huge thx to https://github.com/ruotianluo/pytorch-faster-rcnn/blob/master/lib/nets/resnet_v1.py
    backbone = resnet.resnet50(pretrained=False)
    if pretrained:
        backbone.load_state_dict(model_zoo.load_url(
            'https://s3.us-west-2.amazonaws.com/ai2-rowanz/resnet50-e13db6895d81.th'))
    for i in range(2, 4):
        getattr(backbone, 'layer%d' % i)[0].conv1.stride = (2, 2)
        getattr(backbone, 'layer%d' % i)[0].conv2.stride = (1, 1)
    return backbone


def _load_resnet_imagenet(pretrained=True):
    # huge thx to https://github.com/ruotianluo/pytorch-faster-rcnn/blob/master/lib/nets/resnet_v1.py
    backbone = resnet.resnet50(pretrained=pretrained)
    for i in range(2, 4):
        getattr(backbone, 'layer%d' % i)[0].conv1.stride = (2, 2)
        getattr(backbone, 'layer%d' % i)[0].conv2.stride = (1, 1)
    # use stride 1 for the last conv4 layer (same as tf-faster-rcnn)
    backbone.layer4[0].conv2.stride = (1, 1)
    backbone.layer4[0].downsample[0].stride = (1, 1)

    # # Make batchnorm more sensible
    # for submodule in backbone.modules():
    #     if isinstance(submodule, torch.nn.BatchNorm2d):
    #         submodule.momentum = 0.01

    return backbone


class SimpleDetector(nn.Module):
    def __init__(self, pretrained=True, use_bbox=False, semantic=False, final_dim=1024, layer_norm_epsilon=1e-12):
        """
        :param average_pool: whether or not to average pool the representations
        :param pretrained: Whether we need to load from scratch
        :param semantic: Whether or not we want to introduce the mask and the class label early on (default Yes)
        """
        super(SimpleDetector, self).__init__()

        self.ln_f = LayerNorm(final_dim, eps=layer_norm_epsilon)


        self.semantic = semantic
        if self.semantic:
            self.mask_dims = 32
            self.object_embed = torch.nn.Embedding(num_embeddings=91, embedding_dim=128)
            self.mask_upsample = torch.nn.Conv2d(1, self.mask_dims, kernel_size=3,
                                                 stride=2 if USE_IMAGENET_PRETRAINED else 1,
                                                 padding=1, bias=True)
        else:
            self.object_embed = None
            self.mask_upsample = None

        self.obj_downsample = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(2048 + (128 if semantic else 0), final_dim),
            torch.nn.ReLU(inplace=True),
        )

        self.use_bbox = use_bbox
        if use_bbox:
            self.bbox_upsample = torch.nn.Sequential(
                torch.nn.Dropout(p=0.1),
                torch.nn.Linear(4, final_dim),
                torch.nn.ReLU(inplace=True),
            )
        self.regularizing_predictor = torch.nn.Linear(2048, 91)

    def forward(self,
                img_feats: torch.Tensor,
                boxes: torch.Tensor,
                box_mask: torch.LongTensor,
                obj_labels: torch.LongTensor,
                ):
        """
        :param images: [batch_size, max_num_objects, 2048]
        :param boxes:  [batch_size, max_num_objects, 7] Padded boxes
        :param box_mask: [batch_size, max_num_objects] Mask for whether or not each box is OK
        :return: object reps [batch_size, max_num_objects, dim]
        """

        box_inds = box_mask.nonzero()
        rois = img_feats[box_inds[:, 0], box_inds[:, 1]]
        if self.semantic:
            aligned_obj_labels = obj_labels[box_inds[:, 0], box_inds[:, 1]]
            rois = torch.cat((rois, self.object_embed(aligned_obj_labels)), -1)

        roi_aligned_feats = self.ln_f(self.obj_downsample(rois))

        if self.use_bbox:
            bboxes = boxes[box_inds[:, 0], box_inds[:, 1]]
            box_feats = self.ln_f(self.bbox_upsample(bboxes))
            roi_aligned_feats = roi_aligned_feats + box_feats


        # Add some regularization, encouraging the model to keep giving decent enough predictions
        # obj_logits = self.regularizing_predictor(roi_aligned_feats)
        # obj_labels = classes[box_inds[:, 0], box_inds[:, 1]]
        # cnn_regularization = F.cross_entropy(obj_logits, obj_labels, size_average=True)[None]

        # Reshape into a padded sequence - this is expensive and annoying but easier to implement and debug...
        obj_reps = pad_sequence(roi_aligned_feats, box_mask.sum(1).tolist())

        return {
            'obj_reps': obj_reps,
            # 'obj_logits': obj_logits,
            # 'obj_labels': obj_labels,
            'cnn_regularization_loss': None # cnn_regularization
        }
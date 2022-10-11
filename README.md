# CHOI_inference

Using CHOI for inference with mmdet, ONNX

## PyTorch

### environment

from C-HOI repo

```
python3.7
pytorch=1.5.0 
mmcv=0.4.3
```

### dependency

```
imagecorruptions 1.1.2
convert-onnx-to-caffe2
pycocotools 2.0.5
terminaltables==3.1.10
six==1.16.0
matplotlib==3.3.4
numpy==1.21.6
mmcv==0.2.15
scipy==1.5.4
scikit-image==0.17.2
opencv-python==4.6.0.66
Pillow==9.2.0
future==0.18.2
python-dateutil==2.8.2
pyparsing==3.0.9
kiwisolver==1.3.1
cycler==0.11.0
opencv-python-headless==4.6.0.66
requests==2.27.1
addict==2.4.0
PyYAML==6.0
PyWavelets==1.1.1
tifffile==2020.9.3
imageio==2.15.0
networkx==2.5.1
idna==3.4
charset-normalizer==2.0.12
certifi==2021.5.30
urllib3==1.26.12
decorator==4.4.2
mmdet==1.0rc0+unknown
```

### dataset

[Person in Context](http://picdataset.com:8000/challenge/task/download/)

### Test on HOIW

```
python tools/test_hoiw.py configs/hoiw/cascade_rcnn_x101_64x4d_fpn_1x_4gpu_rel.py hoiw_latest.pth --json_out det_result.json --hoiw_out hoiw_result.json
```

### Test Result

```
{'type': 'HoiwDataset', 'ann_file': 'data/hoiw/annotations/hoiw_test.json', 'rel_ann_file': 'data/hoiw/annotations/test_2019.json', 'img_prefix': 'data/hoiw/test/', 'img_norm_cfg': {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_rgb': True}, 'img_scale': (1200, 700), 'size_divisor': 32, 'flip_ratio': 0, 'with_mask': False, 'with_label': True, 'test_mode': True}
None
{'ann_file': 'data/hoiw/annotations/hoiw_test.json', 'rel_ann_file': 'data/hoiw/annotations/test_2019.json', 'img_prefix': 'data/hoiw/test/', 'img_norm_cfg': {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_rgb': True}, 'img_scale': (1200, 700), 'size_divisor': 32, 'flip_ratio': 0, 'with_mask': False, 'with_label': True, 'test_mode': True}
loading annotations into memory...
Done (t=0.01s)
creating index...
index created!
test_007088.jpg
test_000013.png
test_001329.png
test_001559.png
test_004184.png
test_005012.png
test_006226.png
{'num_stages': 3, 'pretrained': None, 'backbone': {'type': 'ResNeXt', 'depth': 101, 'groups': 64, 'base_width': 4, 'num_stages': 4, 'out_indices': (0, 1, 2, 3), 'frozen_stages': 1, 'style': 'pytorch', 'dcn': {'modulated': False, 'groups': 64, 'deformable_groups': 1, 'fallback_on_stride': False}, 'stage_with_dcn': (False, True, True, True)}, 'neck': {'type': 'FPN', 'in_channels': [256, 512, 1024, 2048], 'out_channels': 256, 'num_outs': 5}, 'rpn_head': {'type': 'RPNHead', 'in_channels': 256, 'feat_channels': 256, 'anchor_scales': [8], 'anchor_ratios': [0.3333333333333333, 0.5, 1.0, 2.0, 3.0], 'anchor_strides': [4, 8, 16, 32, 64], 'target_means': [0.0, 0.0, 0.0, 0.0], 'target_stds': [1.0, 1.0, 1.0, 1.0], 'loss_cls': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'loss_weight': 1.0}, 'loss_bbox': {'type': 'SmoothL1Loss', 'beta': 0.1111111111111111, 'loss_weight': 1.0}}, 'bbox_roi_extractor': {'type': 'SingleRoIExtractor', 'roi_layer': {'type': 'RoIAlign', 'out_size': 7, 'sample_num': 2}, 'out_channels': 256, 'featmap_strides': [4, 8, 16, 32]}, 'bbox_head': [{'type': 'SharedFCBBoxHead', 'num_fcs': 2, 'in_channels': 256, 'fc_out_channels': 1024, 'roi_feat_size': 7, 'num_classes': 12, 'target_means': [0.0, 0.0, 0.0, 0.0], 'target_stds': [0.1, 0.1, 0.2, 0.2], 'reg_class_agnostic': True, 'loss_cls': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0}, 'loss_bbox': {'type': 'SmoothL1Loss', 'beta': 1.0, 'loss_weight': 1.0}}, {'type': 'SharedFCBBoxHead', 'num_fcs': 2, 'in_channels': 256, 'fc_out_channels': 1024, 'roi_feat_size': 7, 'num_classes': 12, 'target_means': [0.0, 0.0, 0.0, 0.0], 'target_stds': [0.05, 0.05, 0.1, 0.1], 'reg_class_agnostic': True, 'loss_cls': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0}, 'loss_bbox': {'type': 'SmoothL1Loss', 'beta': 1.0, 'loss_weight': 1.0}}, {'type': 'SharedFCBBoxHead', 'num_fcs': 2, 'in_channels': 256, 'fc_out_channels': 1024, 'roi_feat_size': 7, 'num_classes': 12, 'target_means': [0.0, 0.0, 0.0, 0.0], 'target_stds': [0.033, 0.033, 0.067, 0.067], 'reg_class_agnostic': True, 'loss_cls': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0}, 'loss_bbox': {'type': 'SmoothL1Loss', 'beta': 1.0, 'loss_weight': 1.0}}], 'rel_head': {'type': 'ReldnHead', 'dim_in': 37632, 'num_prd_classes': 11, 'use_freq_bias': True, 'use_spatial_feat': True, 'add_so_scores': True, 'add_scores_all': False, 'mode': 'hoiw'}, 'train_cfg': None, 'test_cfg': {'rpn': {'nms_across_levels': False, 'nms_pre': 1000, 'nms_post': 1000, 'max_num': 1000, 'nms_thr': 0.7, 'min_bbox_size': 0}, 'rel': {'must_overlap': True, 'run_baseline': False, 'thresh': 0.01}, 'rcnn': {'score_thr': 0.05, 'nms': {'type': 'nms', 'iou_thr': 0.25}, 'max_per_img': 100}, 'keep_all_stages': False}}
{'depth': 101, 'groups': 64, 'base_width': 4, 'num_stages': 4, 'out_indices': (0, 1, 2, 3), 'frozen_stages': 1, 'style': 'pytorch', 'dcn': {'modulated': False, 'groups': 64, 'deformable_groups': 1, 'fallback_on_stride': False}, 'stage_with_dcn': (False, True, True, True)}
{'in_channels': [256, 512, 1024, 2048], 'out_channels': 256, 'num_outs': 5}
{'in_channels': 256, 'feat_channels': 256, 'anchor_scales': [8], 'anchor_ratios': [0.3333333333333333, 0.5, 1.0, 2.0, 3.0], 'anchor_strides': [4, 8, 16, 32, 64], 'target_means': [0.0, 0.0, 0.0, 0.0], 'target_stds': [1.0, 1.0, 1.0, 1.0], 'loss_cls': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'loss_weight': 1.0}, 'loss_bbox': {'type': 'SmoothL1Loss', 'beta': 0.1111111111111111, 'loss_weight': 1.0}}
{'use_sigmoid': True, 'loss_weight': 1.0}
{'beta': 0.1111111111111111, 'loss_weight': 1.0}
{'roi_layer': {'type': 'RoIAlign', 'out_size': 7, 'sample_num': 2}, 'out_channels': 256, 'featmap_strides': [4, 8, 16, 32]}
{'num_fcs': 2, 'in_channels': 256, 'fc_out_channels': 1024, 'roi_feat_size': 7, 'num_classes': 12, 'target_means': [0.0, 0.0, 0.0, 0.0], 'target_stds': [0.1, 0.1, 0.2, 0.2], 'reg_class_agnostic': True, 'loss_cls': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0}, 'loss_bbox': {'type': 'SmoothL1Loss', 'beta': 1.0, 'loss_weight': 1.0}}
{'use_sigmoid': False, 'loss_weight': 1.0}
{'beta': 1.0, 'loss_weight': 1.0}
{'roi_layer': {'type': 'RoIAlign', 'out_size': 7, 'sample_num': 2}, 'out_channels': 256, 'featmap_strides': [4, 8, 16, 32]}
{'num_fcs': 2, 'in_channels': 256, 'fc_out_channels': 1024, 'roi_feat_size': 7, 'num_classes': 12, 'target_means': [0.0, 0.0, 0.0, 0.0], 'target_stds': [0.05, 0.05, 0.1, 0.1], 'reg_class_agnostic': True, 'loss_cls': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0}, 'loss_bbox': {'type': 'SmoothL1Loss', 'beta': 1.0, 'loss_weight': 1.0}}
{'use_sigmoid': False, 'loss_weight': 1.0}
{'beta': 1.0, 'loss_weight': 1.0}
{'roi_layer': {'type': 'RoIAlign', 'out_size': 7, 'sample_num': 2}, 'out_channels': 256, 'featmap_strides': [4, 8, 16, 32]}
{'num_fcs': 2, 'in_channels': 256, 'fc_out_channels': 1024, 'roi_feat_size': 7, 'num_classes': 12, 'target_means': [0.0, 0.0, 0.0, 0.0], 'target_stds': [0.033, 0.033, 0.067, 0.067], 'reg_class_agnostic': True, 'loss_cls': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0}, 'loss_bbox': {'type': 'SmoothL1Loss', 'beta': 1.0, 'loss_weight': 1.0}}
{'use_sigmoid': False, 'loss_weight': 1.0}
{'beta': 1.0, 'loss_weight': 1.0}
{'dim_in': 37632, 'num_prd_classes': 11, 'use_freq_bias': True, 'use_spatial_feat': True, 'add_so_scores': True, 'add_scores_all': False, 'mode': 'hoiw'}
{'use_sigmoid': True, 'loss_weight': 1.0}
The model and loaded state dict do not match exactly

missing keys in source state_dict: sa.theta.conv.weight, sa.phi.conv.bias, sa.conv_out.conv.bias, sa.phi.conv.weight, sa.theta.conv.bias, sa.conv_out.conv.weight, sa.g.conv.weight, sa.g.conv.bias

[                                                  ] 0/8794, elapsed: 0s, ETA:/opt/conda/conda-bld/pytorch_1587428398394/work/torch/csrc/utils/python_arg_parser.cpp:756: UserWarning: This overload of addcmul is deprecated:
        addcmul(Tensor input, Number value, Tensor tensor1, Tensor tensor2, *, Tensor out)
Consider using one of the following signatures instead:
        addcmul(Tensor input, Tensor tensor1, Tensor tensor2, *, Number value, Tensor out)
[>>                                                ] 419/8794, 5.1 task/s, elapsed: 82s, ETA:  1643sPremature end of JPEG file
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ] 8727/8794, 4.5 task/s, elapsed: 1928s, ETA:    15slibpng warning: iCCP: known incorrect sRGB profile
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 8794/8794, 4.5 task/s, elapsed: 1942s, ETA:     0s
real    32m27.400s
user    35m14.424s
sys     5m35.886s
```

### evaluation

```
from evalution import hoiw_eval
# 标注的测试集
eval_demo = hoiw_eval('data/hoiw/annotations/test_2019.json')
# 跑测试集生成的结果
eval_demo.evalution('hoiw_result.json')
```

### result

```
class 1 --- ap: 0.5130739181199342   max recall: 0.631827731092437
class 2 --- ap: 0.8094335485894057   max recall: 0.9116616031783127
class 3 --- ap: 0.44967810317134627   max recall: 0.8385525400139179
class 4 --- ap: 0.28223250674761036   max recall: 0.5300546448087432
class 5 --- ap: 0.8685585453142284   max recall: 0.9372496662216289
class 6 --- ap: 0.8303554790254954   max recall: 0.9577543079488605
class 7 --- ap: 0.7413159590492145   max recall: 0.8287608363080061
class 8 --- ap: 0.6426917935524193   max recall: 0.9052631578947369
class 9 --- ap: 0.38903387376090576   max recall: 0.8711111111111111
class 10 --- ap: 0.18826843638750299   max recall: 0.4748743718592965
--------------------
mAP: 0.5714642163718063   max recall: 0.7887109970437051
--------------------
```

## ONNX

### convert pytorch PTH model to onnx

modify mmcv source code

```
anaconda3/envs/onnx/lib/python3.8/site-packages/mmcv/runner/checkpoint.py
```

### onnxruntime inference

## Q&A

**Q1-[PyTorch]:**

```
ImportError: cannot import name 'deform_conv_cuda' from 'mmdet.ops.dcn' (/ai_app/C-HOI/mmdet/ops/dcn/__init__.py)
```

need to rebuild module

**A1:**

```
cd C-HOI
python setup.py develop 
```

**Q2-[PyTorch]:**

```
Traceback (most recent call last):
  File "tools/test_hoiw.py", line 242, in <module>
    main()
  File "tools/test_hoiw.py", line 174, in main
    dataset = build_dataset(cfg.data.test)
  File "/root/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/mmdet-1.0rc0+unknown-py3.7-linux-x86_64.egg/mmdet/datasets/builder.py", line 37, in build_dataset
    dataset = build_from_cfg(cfg, DATASETS, default_args)
  File "/root/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/mmdet-1.0rc0+unknown-py3.7-linux-x86_64.egg/mmdet/utils/registry.py", line 67, in build_from_cfg
    obj_type, registry.name))
KeyError: 'None is not in the dataset registry'
```

**A2:**

```
use mmdet in C-HOI folder, DO NOT install mmdet seperately  
```

**Q3-[PyTorch]**:

```
error in deformable_im2col: no kernel image is available for execution on the device
```

cuda kernel compute architecture is not compatible with enviroment, need to rebuild

**A3:**

```
cd C-HOI
python setup.py develop 
```

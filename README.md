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

-

### onnxruntime inference

-

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

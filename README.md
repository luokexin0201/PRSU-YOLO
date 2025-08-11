# PRSU-YOLO [![DOI](https://zenodo.org/badge/1033178804.svg)](https://doi.org/10.5281/zenodo.16791499)

This code is directly related to the manuscript(Enhanced Small Object Detection in UAV Imagery:
 PRSU-YOLO with Multi-Scale Adaptability) submitted to the Visual Computer.



## Performance 

VisDrone 2019

| Model | Test Size | Precision<sup>test</sup> | Recall<sub>50<sup>test</sup> | mAP<sub>50</sub><sup>test</sup> | mAP<sub>50:95</sub><sup>test</sup> | GFLOPs |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: |
| **PRSU-YOLO** | 640 | **49.9%** | **37.9%** | **37.4%** | **22.4%** | **53.4** |


## Evaluation

``` shell
# evaluate converted PRSU-YOLO 
python val.py --data data/VisDrone.yaml --img 640 --batch 32 --conf 0.001 --iou 0.7 --device 0 --weights './PRSU.pt' --save-json --name PRSU_640_val

# evaluate yolov9 models
# python val_dual.py --data data/VisDrone.yaml --img 640 --batch 32 --conf 0.001 --iou 0.7 --device 0 --weights './PRSU.pt' --save-json --name PRSU_640_val

```

## Training

Data preparation

* Download VisDrone dataset images (https://github.com/VisDrone/VisDrone-Dataset)(https://github.com/VisDrone/VisDrone-Dataset). 

Single GPU training

``` shell
# train PRSU
python train_dual.py --workers 8 --device 0 --batch 16 --data data/VisDrone.yaml --img 640 --cfg models/detect/PRSU.yaml --weights '' --name PRSU --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15

```

Multiple GPU training

``` shell
# train PRSU
python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train_dual.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch 128 --data data/VisDrone.yaml --img 640 --cfg models/detect/PRSU.yaml --weights '' --name PRSU --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15

```

## Inference

``` shell
# inference converted PRSU
python detect.py --source './data/images/horses.jpg' --img 640 --device 0 --weights './PRSU.pt' --name PRSU_640_detect

# inference PRSU
# python detect_dual.py --source './data/images/horses.jpg' --img 640 --device 0 --weights './PRSU.pt' --name PRSU_640_detect

```


## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
* [https://github.com/VDIGPKU/DynamicDet](https://github.com/VDIGPKU/DynamicDet)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/meituan/YOLOv6](https://github.com/meituan/YOLOv6)

</details>

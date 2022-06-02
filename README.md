# PAC Prediction Sets

## Classification

### MNIST
```
python3 main_cls_mnist.py --exp_name exp_mnist
```

## Detection

### COCO

Assuming the COCO dataset is at `/home/sangdonp/data/coco2017/`, execute the following to initialize the COCO dataset:
```
ln -s /home/sangdonp/data/coco2017/ data/coco
```

To construct prediction sets and evaluation, execute the following:
```
cal_main_det_coco.sh
```

Given constructed prediction sets, the following only evaluates them.
```
run_main_det_coco.sh
```


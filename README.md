# PAC Prediction Sets

## Classification

### MNIST
```
python3 main_cls_mnist.py --exp_name exp_mnist
```

## Detection

### COCO
To construct prediction sets and evaluation, execute the following:
```
cal_main_det_coco.sh
```

Given constructed prediction sets, the following only evaluates them.
```
run_main_det_coco.sh
```


GPUID=1
MODEL_PATH=/data/sangdonp/models/prob_fasterrcnn_coco/model_params_best

CUDA_VISIBLE_DEVICES=$GPUID python3 main_det_coco.py \
		    --exp_name pac_ps_det_coco \
		    --data.src COCO \
		    --model.path_pretrained $MODEL_PATH \
		    --estimate \
		    --estimate_proposal \
		    --estimate_objectness \
		    --estimate_location

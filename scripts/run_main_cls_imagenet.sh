
for n in 15000 20000 25000
do
    for eps in 0.1 0.05 0.01
    do
	for delta in 1e-5 1e-3 1e-1
	do
	    for i in {1..100}
	    do
		python3 main_cls_imagenet.py \
			--exp_name pac_ps_imagenet-n-${n}-eps-${eps//./d}-delta-${delta}-iter-${i} \
			--data.src ImageNet \
			--data.seed None \
			--model.path_pretrained pytorch \
			--train.skip_eval \
			--train_ps.save_compact \
			--train_ps.n $n \
			--train_ps.eps $eps \
			--train_ps.delta $delta
	    done
	done
    done
done

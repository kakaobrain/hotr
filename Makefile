# [V-COCO] single-gpu train (runs in 1 GPU)
single_train:
	python main.py \
		--group_name KakaoBrain \
		--run_name single_run_000001 \
		--HOIDet \
		--validate \
		--share_enc \
		--pretrained_dec \
		--lr 1e-4 \
		--num_hoi_queries 100 \
		--set_cost_idx 10 \
		--hoi_act_loss_coef 10 \
		--hoi_eos_coef 0.1 \
		--temperature 0.1 \
		--no_aux_loss \
		--hoi_aux_loss \
		--dataset_file vcoco \
		--frozen_weights https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
		--data_path v-coco \
		--output_dir checkpoints/vcoco/

# [V-COCO] single-gpu test (runs in 1 GPU)
single_test:
	python main.py \
		--HOIDet \
		--share_enc \
		--pretrained_dec \
		--num_hoi_queries 16 \
		--object_threshold 0 \
		--temperature 0.05 \
		--no_aux_loss \
		--eval \
		--dataset_file vcoco \
		--data_path v-coco \
		--resume checkpoints/vcoco/q16.pth

# [V-COCO] multi-gpu train (runs in 8 GPUs)
multi_train:
	python -m torch.distributed.launch \
		--nproc_per_node=8 \
		--use_env main.py \
		--group_name KakaoBrain \
		--run_name multi_run_000001 \
		--HOIDet \
		--wandb \
		--validate \
		--share_enc \
		--pretrained_dec \
		--lr 1e-4 \
		--num_hoi_queries 16 \
		--set_cost_idx 1 \
		--set_cost_act 1 \
		--hoi_idx_loss_coef 1 \
		--hoi_act_loss_coef 10 \
		--hoi_eos_coef 0.1 \
		--temperature 0.05 \
		--no_aux_loss \
		--hoi_aux_loss \
		--dataset_file vcoco \
		--frozen_weights https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
		--data_path v-coco \
		--output_dir checkpoints/vcoco/

# [V-COCO] multi-gpu test (runs in 8 GPUs)
multi_test:
	python -m torch.distributed.launch \
		--nproc_per_node=8 \
		--use_env main.py \
		--HOIDet \
		--share_enc \
		--pretrained_dec \
		--num_hoi_queries 16 \
		--object_threshold 0 \
		--temperature 0.05 \
		--no_aux_loss \
		--eval \
		--dataset_file vcoco \
		--data_path v-coco \
		--resume checkpoints/vcoco/q16.pth

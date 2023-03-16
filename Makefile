# [V-COCO] single-gpu train (runs in 1 GPU)
vcoco_single_train:
	python main.py \
		--group_name MMH_vcoco \
		--run_name vcoco_single_run \
		--HOIDet \
		--wandb \
		--validate \
		--share_enc \
		--pretrained_dec \
		--lr 1e-4 \
		--num_hoi_queries 16 \
		--set_cost_idx 10 \
		--hoi_act_loss_coef 10 \
		--hoi_eos_coef 0.1 \
		--temperature 0.05 \
		--no_aux_loss \
		--hoi_aux_loss \
		--dataset_file vcoco \
		--frozen_weights https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
		--data_path /gemini/code/v-coco  \
		--output_dir checkpoints/vcoco/

# [V-COCO] multi-gpu train (runs in 4 GPUs)
vcoco_multi_train:
	python -m torch.distributed.launch \
		--nproc_per_node=4 \
		--use_env main.py \
		--group_name MMH_vcoco_use_pos_info \
		--run_name vcoco_multi_run \
		--HOIDet \
		--wandb \
		--validate \
		--share_enc \
		--pretrained_dec \
		--lr 1e-4 \
		--num_hoi_queries 16 \
		--set_cost_idx 10 \
		--set_cost_act 1 \
		--hoi_idx_loss_coef 1 \
		--hoi_act_loss_coef 10 \
		--hoi_eos_coef 0.1 \
		--temperature 0.05 \
		--no_aux_loss \
		--hoi_aux_loss \
		--dataset_file vcoco \
		--frozen_weights https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
		--data_path /gemini/code/v-coco \
		--output_dir checkpoints/vcoco/ \
		--use_pos_info

# [V-COCO] single-gpu test (runs in 1 GPU)
vcoco_single_test:
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
		--resume checkpoints/vcoco/vcoco_q16.pth

# [V-COCO] multi-gpu test (runs in 8 GPUs)
vcoco_multi_test:
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
		--data_path v-coco  \
		--resume checkpoints/vcoco/vcoco_q16.pth

# [HICO-DET] single-gpu train (runs in 1 GPU)
hico_single_train:
	python main.py \
		--group_name KakaoBrain_HOTR_hicodet \
		--run_name hicodet_single_run_000001 \
		--HOIDet \
		--validate \
		--share_enc \
		--pretrained_dec \
		--lr 1e-4 \

		--num_hoi_queries 16 \
		--set_cost_idx 20 \
		--hoi_act_loss_coef 10 \
		--hoi_eos_coef 0.1 \
		--temperature 0.2 \
		--no_aux_loss \
		--hoi_aux_loss \
		--dataset_file hico-det \
		--frozen_weights https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
		--data_path /gemini/data-2/hico_20160224_det \
		--output_dir checkpoints/hico_det/


# [HICO-DET] multi-gpu train (runs in 4 GPUs)
hico_multi_train:
	python -m torch.distributed.launch \
		--nproc_per_node=4 \
		--use_env main.py \
		--group_name MMH \
		--run_name hicodet_multi_run_000001 \
		--HOIDet \
		--wandb \
		--validate \
		--share_enc \
		--pretrained_dec \
		--lr 1e-4 \
		--lr_drop 80 \
		--epochs 100 \
		--num_hoi_queries 16 \
		--set_cost_idx 20 \
		--hoi_act_loss_coef 10 \
		--hoi_eos_coef 0.1 \
		--temperature 0.2 \
		--no_aux_loss \
		--hoi_aux_loss \
		--dataset_file hico-det \
		--frozen_weights https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
		--data_path hico_20160224_det \
		--output_dir checkpoints/hico_det/

# [HICO-DET] single-gpu test
hico_single_test:
	python main.py \
		--HOIDet \
		--share_enc \
		--pretrained_dec \
		--num_hoi_queries 16 \
		--object_threshold 0 \
		--temperature 0.2 \
		--no_aux_loss \
		--eval \
		--dataset_file hico-det \
		--data_path hico_20160224_det \
		--resume checkpoints/hico_det/hico_q16.pth

# [HICO-DET] multi-gpu test (runs in 8 GPUs)
hico_multi_test:
	python -m torch.distributed.launch \
		--nproc_per_node=8 \
		--use_env main.py \
		--HOIDet \
		--share_enc \
		--pretrained_dec \
		--num_hoi_queries 16 \
		--object_threshold 0 \
		--temperature 0.2 \
		--no_aux_loss \
		--eval \
		--dataset_file hico-det \
		--data_path hico_20160224_det \
		--resume checkpoints/hico_det/hico_q16.pth

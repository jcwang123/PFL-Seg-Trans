# ========================= step1: federation =========================#
# ========================= step1: federation =========================#
# ========================= step1: federation =========================#

# FedAVG FPN network #
python scripts/train_fed.py --fl fedavg --net pvtb2_fpn --ver 0 --dataset polyp --gpu 3
python scripts/train_fed.py --fl fedavg --net pvtb0_fpn --ver 0 --dataset polyp --gpu 2
python scripts/train_fed.py --fl fedavg --net resnet50_fpn --ver 0 --dataset polyp --gpu 0
python scripts/train_fed.py --fl fedavg --net resnet18_fpn --ver 0 --dataset polyp --gpu 0

# FedRep FPN network
python scripts/train_fed.py --fl fedrep --net resnet18_fpn --ver 0 --dataset polyp --gpu 0
python scripts/train_fed.py --fl fedrep --net pvtb0_fpn --ver 0 --dataset polyp --gpu 0
python scripts/train_fed.py --fl fedrep --net pvtb2_fpn --ver 0 --dataset polyp --gpu 0
python scripts/train_fed.py --fl fedrep --net resnet50_fpn --ver 0 --dataset polyp --gpu 2
python scripts/train_fed.py --fl fedrep --net pvtb0_fpn --ver 2 --dataset polyp --gpu 3

# FedRepv2 FPN network
python scripts/train_fed.py --fl fedrep --net resnet18_fpn --ver 1 --dataset polyp --gpu 3
python scripts/train_fed.py --fl fedrep --net pvtb0_fpn --ver 1 --dataset polyp --gpu 3

# FedBABU FPN network
python scripts/train_fed.py --fl fedbabu --net resnet18_fpn --ver 0 --dataset polyp --gpu 3
python scripts/train_fed.py --fl fedbabu --net pvtb0_fpn --ver 0 --dataset polyp --gpu 3

# FedBABU Ditto network
python scripts/train_fed.py --fl ditto --net resnet18_fpn --ver 0 --dataset polyp --gpu 0
python scripts/train_fed.py --fl ditto --net pvtb0_fpn --ver 0 --dataset polyp --gpu 0

# FedGKD Ditto network
python scripts/train_fed.py --fl fedgkd --net pvtb0_fpn --ver 0 --dataset polyp --gpu 1

# IOP-FL  network
python scripts/train_fed.py --fl iopfl --net pvtb0_fpn --ver 0 --dataset polyp --gpu 1

# Ours
python scripts/train_fed.py --fl ours_v0 --net pvtb0_fpn --ver 0 --dataset polyp --gpu 0
python scripts/train_fed.py --fl ours_q --net pvtb0_fpn --ver 0 --dataset polyp --gpu 0
python scripts/train_fed.py --fl ours_k --net pvtb0_fpn --ver 0 --dataset polyp --gpu 0
python scripts/train_fed.py --fl ours_qk --net pvtb0_fpn --ver 0 --dataset polyp --gpu 3

python scripts/train_fed.py --fl ours_q --net pvtb0_fpn --ver 1 --dataset polyp --gpu 0 &
python scripts/train_fed.py --fl ours_k --net pvtb0_fpn --ver 1 --dataset polyp --gpu 0 &

python scripts/train_fed.py --fl ours_q_head --net pvtb0_fpn --ver 0 --dataset polyp --gpu 1

# ===================================================================#
# ========================= step2: finetune =========================#
# ===================================================================#
#
python scripts/train.py --fl ditto_ft --net pvtb0_fpn --ver 0 --dataset polyp --load_weight ditto_pvtb0_fpn_0 --max_epoch 30 --base_lr 0.0001 --gpu 0 &
python scripts/train.py --fl fedrep_ft --net pvtb0_fpn --ver 0 --dataset polyp --load_weight fedrep_pvtb0_fpn_0 --max_epoch 30 --base_lr 0.0001 --gpu 1 &
python scripts/train.py --fl fedbabu_ft --net pvtb0_fpn --ver 0 --dataset polyp --load_weight fedbabu_pvtb0_fpn_0 --max_epoch 30 --base_lr 0.0001 --gpu 3 &
python scripts/train.py --fl iopfl_ft --net pvtb0_fpn --ver 0 --dataset polyp --load_weight iopfl_pvtb0_fpn_0 --max_epoch 30 --base_lr 0.0001 --gpu 0 &
#
python scripts/train.py --fl ours_q_ft --net pvtb0_fpn --ver 0 --dataset polyp --load_weight ours_q_pvtb0_fpn_0 --max_epoch 30 --base_lr 0.0001 --gpu 1 &
python scripts/train_ft.py --fl ours_q_bd_ft --net pvtb0_fpn --ver 0 --dataset polyp --load_weight ours_q_pvtb0_fpn_0 --max_epoch 50 --base_lr 0.0001 --gpu 0
#
python scripts/train_ft.py --fl ours_q_bd_ft --net pvtb0_fpn --ver 1 --dataset polyp --load_weight ours_q_pvtb0_fpn_0 --max_epoch 50 --base_lr 0.0001 --gpu 1

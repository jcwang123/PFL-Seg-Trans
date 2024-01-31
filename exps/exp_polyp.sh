PARAM="--dataset polyp --max_epoch 200 --base_lr 0.001"
# ========================= step1: local train =========================#
python scripts/train.py --fl local --net pvtb0_fpn --ver 0 --gpu 1 $PARAM &


# ========================= step1: federation =========================#
# FedAVG FPN network #
python scripts/train_fed.py --fl fedavg --net pvtb0_fpn --ver 0 --gpu 1 $PARAM &
# FedRep FPN network
python scripts/train_fed.py --fl fedrep --net pvtb0_fpn --ver 0 --gpu 1 $PARAM &
# FedBABU FPN network
python scripts/train_fed.py --fl fedbabu --net pvtb0_fpn --ver 0 --gpu 2 $PARAM &
# Ditto FPN network
python scripts/train_fed.py --fl ditto --net pvtb0_fpn --ver 0 --gpu 2 $PARAM &
# FedGKD FPN network
python scripts/train_fed.py --fl fedgkd --net pvtb0_fpn --ver 0 --gpu 2 $PARAM &
# IOP-FL  network
python scripts/train_fed.py --fl iopfl --net pvtb0_fpn --ver 0 --gpu 3 $PARAM &
# FedLC  network
python scripts/train_fed.py --fl fedlc --net pvtb0_fpn --ver 0 --gpu 0 $PARAM &
# Ours
python scripts/train_fed.py --fl ours_q --net pvtb0_fpn --ver 0 --gpu 3 $PARAM &

PARAM="--dataset polyp --max_epoch 50 --base_lr 0.0001"
python scripts/train_ft.py --fl fedavg_ft --load_weight fedavg_0_pvtb0_fpn_0 --net pvtb0_fpn --ver 0 --gpu 1 $PARAM 

python scripts/train_ft.py --fl fedlc_ft --load_weight fedlc_0_pvtb0_fpn_0 --net pvtb0_fpn --ver 0 --gpu 1 $PARAM 




# ========================= test =========================#
# ========================= test =====================


# ========================= test =========================#
# # ========================= test =========================#

python scripts/test.py --fl fedavg --net resnet18_fpn --ver 0 --gpu 2 --dataset polyp
python scripts/test.py --fl fedavg_ft_0 --net pvtb0_fpn --ver 0 --gpu 2 --dataset polyp


python scripts/test.py --fl fedavg_ft_0 --net pvtb0_fpn --ver 0 --gpu 2 --dataset polyp

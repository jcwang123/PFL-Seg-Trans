#  local

# ========================= FedAVG FPN network =========================#
python scripts/train_fed.py --fl fedavg --net pvtb2_fpn --ver 0 --dataset polyp --gpu 3
python scripts/train_fed.py --fl fedavg --net pvtb0_fpn --ver 0 --dataset polyp --gpu 2
python scripts/train_fed.py --fl fedavg --net resnet50_fpn --ver 0 --dataset polyp --gpu 0
python scripts/train_fed.py --fl fedavg --net resnet18_fpn --ver 0 --dataset polyp --gpu 0

# ========================= FedRep FPN network =========================#
python scripts/train_fed.py --fl fedrep --net resnet18_fpn --ver 0 --dataset polyp --gpu 0
python scripts/train_fed.py --fl fedrep --net pvtb0_fpn --ver 0 --dataset polyp --gpu 0
python scripts/train_fed.py --fl fedrep --net pvtb2_fpn --ver 0 --dataset polyp --gpu 0
python scripts/train_fed.py --fl fedrep --net resnet50_fpn --ver 0 --dataset polyp --gpu 2

python scripts/train_fed.py --fl fedrep --net pvtb0_fpn --ver 2 --dataset polyp --gpu 3

# ========================= FedRepv2 FPN network =========================#
python scripts/train_fed.py --fl fedrep --net resnet18_fpn --ver 1 --dataset polyp --gpu 3
python scripts/train_fed.py --fl fedrep --net pvtb0_fpn --ver 1 --dataset polyp --gpu 3

# ========================= FedBABU FPN network =========================#
python scripts/train_fed.py --fl fedbabu --net resnet18_fpn --ver 0 --dataset polyp --gpu 3
python scripts/train_fed.py --fl fedbabu --net pvtb0_fpn --ver 0 --dataset polyp --gpu 3

# ========================= FedBABU Ditto network =========================#
python scripts/train_fed.py --fl ditto --net resnet18_fpn --ver 0 --dataset polyp --gpu 0
python scripts/train_fed.py --fl ditto --net pvtb0_fpn --ver 0 --dataset polyp --gpu 0

# ========================= FedGKD Ditto network =========================#
python scripts/train_fed.py --fl fedgkd --net pvtb0_fpn --ver 0 --dataset polyp --gpu 1

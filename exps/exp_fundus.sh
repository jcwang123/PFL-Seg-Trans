# ------------------------------basic federation ------------------------------#
# python scripts/train.py --fl local --net pvtb0_fpn --ver 1 --dataset fundus --max_epoch 100 --base_lr 0.0001 --gpu 1
# python scripts/train.py --fl local --net pvtb0_fpn --ver 0 --dataset fundus --max_epoch 100 --base_lr 0.0001 --gpu 1
# python scripts/train_fed.py --fl fedavg --net pvtb0_fpn --ver 0 --dataset fundus --base_lr 0.0001 --gpu 1
# python scripts/train_fed.py --fl fedrep --net pvtb0_fpn --ver 0 --dataset fundus --base_lr 0.0001 --gpu 1
# python scripts/train_fed.py --fl fedbabu --net pvtb0_fpn --ver 0 --dataset fundus --base_lr 0.0001 --gpu 1
# python scripts/train_fed.py --fl ditto --net pvtb0_fpn --ver 0 --dataset fundus --base_lr 0.0001 --gpu 1
# python scripts/train_fed.py --fl fedgkd --net pvtb0_fpn --ver 0 --dataset fundus --base_lr 0.0001 --gpu 1
# python scripts/train_fed.py --fl iopfl --net pvtb0_fpn --ver 0 --dataset fundus --base_lr 0.0001 --gpu 1
# python scripts/train_fed.py --fl ours_q --net pvtb0_fpn --ver 1 --dataset fundus --base_lr 0.0001 --gpu 1 --max_epoch 400

# python scripts/test.py --fl local --net pvtb0_fpn --ver 0 --dataset fundus --gpu 0
# python scripts/test.py --fl local --net pvtb0_fpn --ver 1 --dataset fundus --gpu 0
# python scripts/test.py --fl fedavg --net pvtb0_fpn --ver 0 --dataset fundus --gpu 0
# python scripts/test.py --fl ditto --net pvtb0_fpn --ver 0 --dataset fundus --gpu 0
# python scripts/test.py --fl fedrep --net pvtb0_fpn --ver 0 --dataset fundus --gpu 0
# python scripts/test.py --fl fedbabu --net pvtb0_fpn --ver 0 --dataset fundus --gpu 0
# python scripts/test.py --fl fedgkd --net pvtb0_fpn --ver 0 --dataset fundus --gpu 0
# python scripts/test.py --fl iopfl --net pvtb0_fpn --ver 0 --dataset fundus --gpu 0
# python scripts/test.py --fl ours_q --net pvtb0_fpn --ver 0 --dataset fundus --gpu 0

# ------------------------------ finetune ------------------------------#
python scripts/train.py --fl ditto_ft --net pvtb0_fpn --ver 0 --dataset fundus --load_weight ditto_pvtb0_fpn_0 --max_epoch 30 --base_lr 0.0001 --gpu 1
python scripts/train.py --fl fedrep_ft --net pvtb0_fpn --ver 0 --dataset fundus --load_weight fedrep_pvtb0_fpn_0 --max_epoch 30 --base_lr 0.0001 --gpu 1
python scripts/train.py --fl fedbabu_ft --net pvtb0_fpn --ver 0 --dataset fundus --load_weight fedbabu_pvtb0_fpn_0 --max_epoch 30 --base_lr 0.0001 --gpu 1
python scripts/train.py --fl iopfl_ft --net pvtb0_fpn --ver 0 --dataset fundus --load_weight iopfl_pvtb0_fpn_0 --max_epoch 30 --base_lr 0.0001 --gpu 1
python scripts/train.py --fl ours_q_ft --net pvtb0_fpn --ver 0 --dataset fundus --load_weight ours_q_pvtb0_fpn_0 --max_epoch 30 --base_lr 0.0001 --gpu 1

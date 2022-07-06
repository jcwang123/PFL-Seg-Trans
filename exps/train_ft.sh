python scripts/train.py --fl ditto_ft --net pvtb0_fpn --ver 0 --dataset polyp --load_weight ditto_pvtb0_fpn_0 --max_epoch 30 --base_lr 0.0001 --gpu 0 &
python scripts/train.py --fl fedrep_ft --net pvtb0_fpn --ver 0 --dataset polyp --load_weight fedrep_pvtb0_fpn_0 --max_epoch 30 --base_lr 0.0001 --gpu 1 &
python scripts/train.py --fl fedbabu_ft --net pvtb0_fpn --ver 0 --dataset polyp --load_weight fedbabu_pvtb0_fpn_0 --max_epoch 30 --base_lr 0.0001 --gpu 3 &
python scripts/train.py --fl iopfl_ft --net pvtb0_fpn --ver 0 --dataset polyp --load_weight iopfl_pvtb0_fpn_0 --max_epoch 20 --base_lr 0.0001 --gpu 0 &

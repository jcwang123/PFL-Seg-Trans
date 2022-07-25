# python scripts/test.py --fl local --net pvtb0_fpn --ver 0 --dataset polyp --gpu 0
# python scripts/test.py --fl fedavg --net pvtb0_fpn --ver 0 --dataset polyp --gpu 0
# python scripts/test.py --fl ditto --net pvtb0_fpn --ver 0 --dataset polyp --gpu 0
# python scripts/test.py --fl fedrep --net pvtb0_fpn --ver 0 --dataset polyp --gpu 0
# python scripts/test.py --fl fedbabu --net pvtb0_fpn --ver 0 --dataset polyp --gpu 0
# python scripts/test.py --fl fedgkd --net pvtb0_fpn --ver 0 --dataset polyp --gpu 0
# python scripts/test.py --fl iopfl --net pvtb0_fpn --ver 0 --dataset polyp --gpu 0

#--------------------------------- finetune ---------------------------------#
# python scripts/test.py --fl fedbabu_ft --net pvtb0_fpn --ver 0 --dataset polyp --gpu 0
# python scripts/test.py --fl iopfl_ft --net pvtb0_fpn --ver 0 --dataset polyp --gpu 0
# python scripts/test.py --fl ditto_ft --net pvtb0_fpn --ver 0 --dataset polyp --gpu 0
# python scripts/test.py --fl fedrep_ft --net pvtb0_fpn --ver 0 --dataset polyp --gpu 0

#--------------------------------- finetune ---------------------------------#
python scripts/test.py --fl ours_q --net pvtb0_fpn --ver 0 --dataset polyp --gpu 2
python scripts/test.py --fl ours_k --net pvtb0_fpn --ver 0 --dataset polyp --gpu 2
python scripts/test.py --fl ours_q --net pvtb0_fpn --ver 1 --dataset polyp --gpu 2
python scripts/test.py --fl ours_k --net pvtb0_fpn --ver 1 --dataset polyp --gpu 2
python scripts/test.py --fl ours_q_head --net pvtb0_fpn --ver 0 --dataset polyp --gpu 2

python scripts/test.py --fl ours_q_ft --net pvtb0_fpn --ver 0 --dataset polyp --gpu 2

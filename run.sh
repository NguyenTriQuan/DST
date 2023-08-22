python main.py --sparse --amp --wandb --seed 18 --sparse_init ERK --method npb --alpha 0.01 --beta 0.02 --l2 5.0e-4 --multiplier 1 --lr 0.1 --lr_score 1 --density 0.01 --update_frequency 1500 --epochs 250 --model ResNet18 --data cifar10 --decay_frequency 30000 --batch-size 128 --growth random --death NPB --redistribution none
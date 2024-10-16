#!/bin/bash  
  
a=(1 10000 20000 30000 40000 50000 60000 70000 80000 84600)  
  
for i in "${a[@]}"; do  
    python main.py --config cylinder_re3900_UNet.yml --seed 1234 --sample_step 1 --t 240 --r 30 --ckpt_path "./train_ddpm/experiments/cylinder_re3900_UNet/logs/ckpt_${i}.pth"  --log_dir "./experiments/cylinder_re3900_UNet_${i}/"
done

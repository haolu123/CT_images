#!/bin/bash

# Local command
module load cuda11.2.2/toolkit
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="172.21.7.53" --master_port=1234 main.py --local_rank=0 --distributed &

# Remote command via SSH
ssh haolu@172.21.7.54 << EOF
module load cuda11.2.2/toolkit
conda activate CT_py39_matplot
cd /isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project2_CT_image/codes/MultiGPU_framework
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr="172.21.7.53" --master_port=1234 main.py --local_rank=0 --distributed
EOF

# Wait for all background processes to finish
wait
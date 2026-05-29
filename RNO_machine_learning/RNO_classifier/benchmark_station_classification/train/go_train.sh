source ~/career/software/venv_torch/bin/activate
# python step2_train.py \
#     --signal /scratch/users/data/baclark/nu.hdf5 \
#     --noise /scratch/users/data/baclark/noise.hdf5 \
#     --seed 265 --epochs 600 --workers 16 --batch-size 256 --lr 1e-3 --arch fpga \
#     --out runs/exp01

python step3_evaluate.py --checkpoint runs/exp01/best_model.pt \
    --signal /scratch/users/data/baclark/nu.hdf5 \
    --noise /scratch/users/data/baclark/noise.hdf5

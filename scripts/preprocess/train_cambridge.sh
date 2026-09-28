for scene in KingsCollege OldHospital ShopFacade StMarysChurch GreatCourt; do
    python train.py \
        -s "datasets/cambridge/${scene}" \
        -m "datasets/3dgs/cambridge/${scene}" \
        -r 1 \
        --iterations 30000 \
        --data_device cpu \
        --densify_grad_threshold 0.0004 \
        --images "processed" \
        --position_lr_init 0.000016 \
        --scaling_lr 0.001
done
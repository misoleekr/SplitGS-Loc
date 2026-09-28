for scene in chess fire heads office pumpkin redkitchen stairs; do
    python train.py \
        -s "datasets/7scenes/${scene}" \
        -m "datasets/3dgs/7scenes/${scene}" \
        --iterations 30000 \
        --data_device cpu \
        --images  ""
done
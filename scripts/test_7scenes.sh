if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <EXP_NAME> <GPU_ID>"
    exit 1
else
    EXP_NAME=$1
    GPU=$2
fi
for scene in chess fire heads office pumpkin redkitchen stairs; do
    python test.py \
        -m datasets/3dgs/7scenes/${scene} \
        --cfg configs/SplitGSLoc.yaml \
        --prefix ${EXP_NAME} \
        --gs_path results/${EXP_NAME}/${scene}/landmark_gs.ply \
        --gpus ${GPU}
done
python utils/calculate_recall.py --result_path results/${EXP_NAME} | tee results/${EXP_NAME}/recall.txt
python utils/calculate_median.py --result_path results/${EXP_NAME} | tee results/${EXP_NAME}/median.txt

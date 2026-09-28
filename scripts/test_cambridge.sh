if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <EXP_NAME> <GPU_ID>"
    exit 1
else
    EXP_NAME=$1
    GPU=$2
fi
for scene in KingsCollege OldHospital ShopFacade StMarysChurch GreatCourt; do
    python test.py \
        -m datasets/3dgs/cambridge/${scene} \
        --cfg configs/SplitGSLoc.yaml \
        --prefix ${EXP_NAME} \
        --gs_path results/${EXP_NAME}/${scene}/landmark_gs.ply \
        --gpus ${GPU}
done
python utils/calculate_recall.py --result_path results/${EXP_NAME} | tee results/${EXP_NAME}/recall.txt
python utils/calculate_median.py --result_path results/${EXP_NAME} | tee results/${EXP_NAME}/median.txt

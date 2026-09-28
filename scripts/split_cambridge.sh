if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <EXP_NAME> <GPU_ID>"
    exit 1
else
    EXP_NAME=$1
    GPU=$2
fi
BETA=1.4
W_THRES=0.1
V_THRES=1e-9
MAX_SPLIT=1
NUM_GS=32768
for scene in KingsCollege OldHospital ShopFacade StMarysChurch GreatCourt; do
    MAP_PATH="datasets/3dgs/cambridge/${scene}"
    SAVE_DIR="results/${EXP_NAME}/${scene}"
    mkdir -p ${SAVE_DIR}
    python split.py \
        -s datasets/cambridge/${scene} \
        -m ${MAP_PATH} \
        -r 1 \
        -f "sp" \
        --save_dir ${SAVE_DIR} \
        --side_beta ${BETA} \
        --weight_threshold ${W_THRES} \
        --volume_threshold ${V_THRES} \
        --max_split ${MAX_SPLIT} \
        --num_landmark ${NUM_GS} \
        --data_device cpu \
        --images "processed" \
        --gpus ${GPU}
done
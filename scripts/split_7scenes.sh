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
for scene in chess fire heads office pumpkin redkitchen stairs; do
    MAP_PATH="datasets/3dgs/7scenes/${scene}"
    SAVE_DIR="results/${EXP_NAME}/${scene}"
    mkdir -p ${SAVE_DIR}
    python split.py \
        -s datasets/7scenes/${scene} \
        -m ${MAP_PATH} \
        -f "sp" \
        --save_dir ${SAVE_DIR} \
        --side_beta ${BETA} \
        --weight_threshold ${W_THRES} \
        --volume_threshold ${V_THRES} \
        --max_split ${MAX_SPLIT} \
        --num_landmark ${NUM_GS} \
        --data_device cpu \
        --images "" \
        --gpus ${GPU}
done

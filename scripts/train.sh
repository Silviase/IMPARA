# please configure the following paths

# ROOT=
# TRAIN_DATA_ROOT=
# TRAIN_DATA=
# MODEL_ROOT=
# LOG=

echo "TRAIN_DATA: ${TRAIN_DATA}"
echo "LOG: ${LOG}"

mkdir -p "${MODEL_ROOT}/${DATA_NAME}/impact"
mkdir -p "${LOG}"

# you can change paths if needed
for lr in "1e-5"
do
python src/train_qe.py \
    --train_dataset "${TRAIN_DATA}/impact/train.pkl" \
    --test_dataset "${TRAIN_DATA}/impact/valid.pkl" \
    --model_dir "${MODEL_ROOT}/${DATA_NAME}/impact" \
    --task_name "2023_02_09 training test" \
    --learning_rate ${lr} \
    --batch_size 16 \
    --num_epochs 10 \
    --lang "en" \
    > ${LOG}/log_impact.out \
    2> ${LOG}/log_impact.err
done

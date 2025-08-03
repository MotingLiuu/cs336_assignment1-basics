RUN_NAME="train_experiment_config2_$(date +%Y-%m-%d_%H-%M-%S)"

LOG_STEP=10
CHECKPOINT_SAVE_STEP=500
BATCH_SIZE=256
NUM_BATCHES=5000
CHECKPOINT_PATH="" 
CHECKPOINT_FOLDER="./checkpoints1"
DEVICE="cuda"

TRAIN_DATA="../tiny_train_tokens.npy"
VALID_DATA="../tiny_valid_tokens.npy"

D_MODEL=512
D_FF=1344
N_LAYERS=4
N_HEADS=16
VOCAB_SIZE=10000
MAX_SEQ_LEN=256
THETA=10000.0

LR=1e-3
WEIGHT_DECAY=0.01
BETAS_1=0.9
BETAS_2=0.999
EPS=1e-8
OPTIMIZER_TYPE="adamw"
NUM_ITERS=1000

LR_SCHEDULER="cosine"
WARMUP_STEPS=100
MAX_LR=1e-3
MIN_LR=0.0
COSINE_ANNEALING_STEPS=5000

python train.py \
    --log_step $LOG_STEP \
    --checkpoint_save_step $CHECKPOINT_SAVE_STEP \
    --batch_size $BATCH_SIZE \
    --num_batches $NUM_BATCHES \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --checkpoint_folder "$CHECKPOINT_FOLDER" \
    --device "$DEVICE" \
    --train_data "$TRAIN_DATA" \
    --valid_data "$VALID_DATA" \
    --d_model $D_MODEL \
    --d_ff $D_FF \
    --n_layers $N_LAYERS \
    --n_heads $N_HEADS \
    --vocab_size $VOCAB_SIZE \
    --max_seq_len $MAX_SEQ_LEN \
    --theta $THETA \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --betas $BETAS_1 $BETAS_2 \
    --eps $EPS \
    --optimizer_type "$OPTIMIZER_TYPE" \
    --num_iters $NUM_ITERS \
    --lr_scheduler "$LR_SCHEDULER" \
    --warmup_steps $WARMUP_STEPS \
    --max_lr $MAX_LR \
    --min_lr $MIN_LR \
    --cosine_annealing_steps $COSINE_ANNEALING_STEPS \
    --run_name "$RUN_NAME"
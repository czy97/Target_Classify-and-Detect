STORE_PARAM_NAME='bestParam.pth'
LOSS_TYPE='SmoothL1Loss' #choose from L1 , L2 and SmoothL1Loss
UPDATE_RULE='sgd' #choose from Adam and sgd
MODEL_TYPE='Vgg' #choose from Vgg and Alex
LOSS_RATIO=1
LR_DECAY=0.95
DATA_AUG=1 #choose from 0 and 1,0:no 1:yes

LOG_STORE_PATH='./LOG/'
PARAM_STORE_PATH='./storedModels/'
STORE_PARAM_NAME=$PARAM_STORE_PATH$MODEL_TYPE-$LOSS_TYPE-$UPDATE_RULE-loss_ratio:$LOSS_RATIO-$LR_DECAY-dataAug:$DATA_AUG-$STORE_PARAM_NAME


BATCH_SIZE=16
NUM_EPOCHS=800
GPU_ID=1

LOG_NAME=$LOG_STORE_PATH$MODEL_TYPE-$BATCH_SIZE-$NUM_EPOCHS-$LOSS_TYPE-$UPDATE_RULE-loss_ratio:$LOSS_RATIO-$LR_DECAY-dataAug:$DATA_AUG.log

python -m train \
--num_epochs $NUM_EPOCHS \
--batchsize $BATCH_SIZE \
--num_workers 4 \
--storeParamName $STORE_PARAM_NAME \
--gpu_id $GPU_ID \
--loss_type $LOSS_TYPE \
--update_rule $UPDATE_RULE \
--model_type $MODEL_TYPE \
--loss_ratio $LOSS_RATIO \
--lr_decay $LR_DECAY \
--data_aug $DATA_AUG \
| tee -a $LOG_NAME





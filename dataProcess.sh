LOGFN=$1


cat $LOGFN | grep 'train Loss' | awk -F ' ' '{print $3}' > $LOGFN.train.loss
cat $LOGFN | grep 'train Loss' | awk -F ' ' '{print $5}' > $LOGFN.train.loss1
cat $LOGFN | grep 'train Loss' | awk -F ' ' '{print $7}' > $LOGFN.train.loss2


cat $LOGFN | grep 'train Acc' | awk -F ' ' '{print $3}' > $LOGFN.train.acc
cat $LOGFN | grep 'train Acc' | awk -F ' ' '{print $5}' > $LOGFN.train.acc1
cat $LOGFN | grep 'train Acc' | awk -F ' ' '{print $7}' > $LOGFN.train.acc2



cat $LOGFN | grep 'val Loss' | awk -F ' ' '{print $3}' > $LOGFN.val.loss
cat $LOGFN | grep 'val Loss' | awk -F ' ' '{print $5}' > $LOGFN.val.loss1
cat $LOGFN | grep 'val Loss' | awk -F ' ' '{print $7}' > $LOGFN.val.loss2


cat $LOGFN | grep 'val Acc' | awk -F ' ' '{print $3}' > $LOGFN.val.acc
cat $LOGFN | grep 'val Acc' | awk -F ' ' '{print $5}' > $LOGFN.val.acc1
cat $LOGFN | grep 'val Acc' | awk -F ' ' '{print $7}' > $LOGFN.val.acc2


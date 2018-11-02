from __future__ import print_function
from __future__ import division
import time
import torch
import copy


#cal IOU between A and B
#A and B are all torch tensor
def IOU(A,B):
    '''
    LT:left top
    RB:right bottom
    '''
    A_LT_X = A[0].item()
    A_LT_Y = A[1].item()
    A_RB_X = A[0].item()+A[2].item()
    A_RB_Y = A[1].item()+A[3].item()

    B_LT_X = B[0].item()
    B_LT_Y = B[1].item()
    B_RB_X = B[0].item() + B[2].item()
    B_RB_Y = B[1].item() + B[3].item()

    W = min(A_RB_X, B_RB_X) - max(A_LT_X, B_LT_X)
    H = min(A_RB_Y, B_RB_Y) - max(A_LT_Y, B_LT_Y)
    if W <= 0 or H <= 0:
        return 0.0
    SA = A[2].item() * A[3].item()
    SB = B[2].item() * B[3].item()
    cross = W * H
    return cross/(SA + SB - cross)
#cal IOU between batch data predict_bbox and bbox
#if the IOU is greater than thres,the res of corresponding index will be True,means it is a correct regression
def batchIOU_correct(predict_bbox,bbox,thres = 0.5,device = None):
    N,M = predict_bbox.size()
    resTensor = torch.zeros(N)

    for i in range(N):
        resTensor[i] = IOU(predict_bbox[i][:],bbox[i][:])

    return (resTensor > thres).to(device)



def train_model(model, dataloaders, criterion, optimizer, num_epochs=25,device = None,loss_ratio = 1.0,lr_decay = 1.0):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=lr_decay)
    for epoch in range(num_epochs):

        scheduler.step()

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_loss1 = 0.0
            running_loss2 = 0.0
            running_classify_correct = 0
            running_bbox_correct = 0
            running_corrects = 0

            # Iterate over data.
            for val in dataloaders[phase]:
                inputs = val['image'].to(device)
                labels = val['label'].to(device)
                bbox = val['bbox'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if  phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        output_label, output_bbox = model(inputs)
                        loss1 = criterion[0](output_label, labels)
                        loss2 = criterion[1](output_bbox, bbox)
                        loss = 0.5*loss1 + 0.5*loss2* loss_ratio
                    else:
                        output_label, output_bbox = model(inputs)
                        loss1 = criterion[0](output_label, labels)
                        loss2 = criterion[1](output_bbox, bbox)
                        loss = 0.5 * loss1 + 0.5 * loss2 * loss_ratio

                    _, preds = torch.max(output_label, 1)
                    bboxPred = batchIOU_correct(output_bbox,bbox,thres = 0.5,device = device)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_loss1 += loss1.item() * inputs.size(0)
                running_loss2 += loss2.item() * inputs.size(0)

                res_tmp = (preds == labels.data)
                running_classify_correct += torch.sum(res_tmp)
                running_bbox_correct += torch.sum(bboxPred)
                running_corrects += torch.sum(res_tmp*bboxPred)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_loss1 = running_loss1 / len(dataloaders[phase].dataset)
            epoch_loss2 = running_loss2 / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_acc_classify = running_classify_correct.double() / len(dataloaders[phase].dataset)
            epoch_acc_bbox = running_bbox_correct.double() / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Loss_classify: {:.4f} Loss_bbox: {:.4f}'.format(phase, epoch_loss, epoch_loss1,epoch_loss2))
            print('{} Acc: {:.4f} Acc_classify: {:.4f} Acc_bbox: {:.4f}'.format(phase,epoch_acc,epoch_acc_classify,epoch_acc_bbox))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        for param_group in optimizer.param_groups:
            print('Now the lr is {}'.format(param_group['lr']))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

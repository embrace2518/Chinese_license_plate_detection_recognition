from __future__ import absolute_import
import time
import torch.nn.functional as F
import cv2
from torch._dynamo.polyfills import os
import lib.utils.utils as utils
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(config, train_loader, dataset, converter, model, criterion, optimizer, device, epoch, writer_dict=None,
          output_dict=None):
    rec_result_dir = "D:/datasets/images/rec_result"
    batch_time = AverageMeter()  # 记录每批训练耗时
    data_time = AverageMeter()  # 记录数据加载耗时
    losses = AverageMeter()  # 记录损失值变化

    model.train()

    end = time.time()
    for i, (inp, idx) in enumerate(train_loader):
        # measure data time
        data_time.update(time.time() - end)
        labels = utils.get_batch_label(dataset, idx)
        inp = inp.to(device)
        # 修改后：同时接收主输出和汉字分支输出
        main_preds, chinese_preds = model(inp)
        preds = main_preds.cpu()
        chinese_preds = chinese_preds.cpu()
        batch_size = inp.size(0)
        text, length = converter.encode(labels)  # length = 一个batch中的总字符长度, text = 一个batch中的字符所对应的下标
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)  # timestep * batchsize
        # 添加维度验证断言
        assert preds_size.dim() == 1, f"preds_size应为1维，实际维度{preds_size.dim()}"
        assert preds_size.size(0) == batch_size, f"preds_size长度{preds_size.size(0)}应与batch_size{batch_size}一致"
        # 汉字分支损失
        chinese_texts = [label[0] for label in labels]  # 提取所有样本的汉字标签
        _, chinese_labels = converter.encode(chinese_texts)  # 使用相同的转换器编码
        chinese_labels = chinese_labels.cpu().long()  # 添加.long()转换
        chinese_loss = F.nll_loss(chinese_preds, chinese_labels)

        main_loss = criterion(preds, text, preds_size, length)
        loss = main_loss + 0.5 * chinese_loss  # 调整权重系数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), inp.size(0))
        batch_time.update(time.time() - end)

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                speed=inp.size(0) / batch_time.val,
                data_time=data_time, loss=losses)
            print(msg)

            # with torch.no_grad():
            #     # 确保维度正确 (seq_len, batch_size, num_classes)
            #     preds_perm = preds.permute(1, 0, 2) if preds.dim() == 3 else preds.unsqueeze(0)
            #     _, preds_idx = preds_perm.max(2)
            #     # 计算实际序列长度
            #     actual_seq_len = preds_perm.size(0)
            #     preds_size = torch.IntTensor([actual_seq_len] * batch_size)
            #     preds_str = converter.decode(preds_idx.data, preds_size, raw=False)
            #
            # # 保存前4张图像
            # for j in range(min(4, inp.size(0))):
            #     img = inp[j].cpu().numpy()
            #     img = img * config.DATASET.STD + config.DATASET.MEAN  # 反归一化
            #     img = (img * 255).transpose(1, 2, 0).astype('uint8')
            #     label = labels[j]
            #     pred = preds_str[j]
            #     cv2.putText(img, f'GT:{label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            #     cv2.putText(img, f'Pred:{pred}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            #     cv2.imwrite(os.path.join(rec_result_dir, f'epoch{epoch}_batch{i}_sample{j}.jpg'), img)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.avg, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1


def validate(config, val_loader, dataset, converter, model, criterion, device, epoch, writer_dict, output_dict):
    losses = AverageMeter()
    model.eval()
    n_correct = 0
    sum = 0
    with torch.no_grad():
        for i, (inp, idx) in enumerate(val_loader):
            labels = utils.get_batch_label(dataset, idx)
            inp = inp.to(device)
            # inference
            main_preds, _ = model(inp)
            preds = main_preds.cpu()
            batch_size = inp.size(0)
            text, length = converter.encode(labels)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            loss = criterion(preds, text, preds_size, length)
            losses.update(loss.item(), inp.size(0))
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            for pred, target in zip(sim_preds, labels):
                sum += 1
                if pred == target:
                    n_correct += 1

            if (i + 1) % config.PRINT_FREQ == 0:
                print('Epoch: [{0}][{1}/{2}]'.format(epoch, i, len(val_loader)))

            if i == config.TEST.NUM_TEST:
                break

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:config.TEST.NUM_TEST_DISP]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, labels):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    print(n_correct)
    print(config.TEST.NUM_TEST * config.TEST.BATCH_SIZE_PER_GPU)
    # accuracy = n_correct / float(config.TEST.NUM_TEST * config.TEST.BATCH_SIZE_PER_GPU)
    accuracy = n_correct / sum
    print('Test loss: {:.4f}, accuray: {:.4f}'.format(losses.avg, accuracy))

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_acc', accuracy, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return accuracy

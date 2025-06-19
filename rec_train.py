import argparse
from easydict import EasyDict as edict
import yaml
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import lib.utils.utils as utils
from lib.core import function
from lib.dataset._360cc import _360CC
from lib.dataset._own import _OWN
from lib.utils.utils import model_info
from rec_plateNet import myNet_ocr
from rec_alphabets import plateName, plate_chr


# 命令行参数解析器(ArgumentParser)的作用，顾名思义，就是从命令行获取参数。
def parse_arg():
    parser = argparse.ArgumentParser(description="train crnn")

    parser.add_argument('--cfg', help='experiment configuration filename', required=True, type=str)
    parser.add_argument('--img_h', type=int, default=48, help='height')
    parser.add_argument('--img_w', type=int, default=168, help='width')
    args = parser.parse_args()

    with open(args.cfg, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # config = yaml.load(f)
        config = edict(config)  # 将普通的字典对象转换为支持属性访问的特殊字典

    config.DATASET.ALPHABETS = plateName
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)
    config.HEIGHT = args.img_h
    config.WIDTH = args.img_w
    return config


def get_dataset(config):
    if config.DATASET.DATASET == "360CC":
        return _360CC
    elif config.DATASET.DATASET == "OWN":
        return _OWN
    else:
        raise NotImplemented()


def main():
    # load config
    config = parse_arg()

    # 检查点及日志
    output_dict = utils.create_log_folder(config, phase='train')

    # cudnn
    # 启用自动寻找最优卷积算法（当输入尺寸固定时加速训练）
    cudnn.benchmark = config.CUDNN.BENCHMARK
    # 确保可重复性（禁用非确定性算法）
    cudnn.deterministic = config.CUDNN.DETERMINISTIC  # 调试时设为True，生产环境设为False
    # 全局启用/禁用cuDNN加速
    cudnn.enabled = config.CUDNN.ENABLED

    # writer dict
    writer_dict = {
        'writer': SummaryWriter(log_dir=output_dict['tb_dir']),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # construct face related neural networks
    # cfg =[8,8,16,16,'M',32,32,'M',48,48,'M',64,128] #small model
    cfg = [16, 16, 32, 32, 'M', 64, 64, 'M', 96, 96, 'M', 128, 128]  # medium model
    # cfg = [32, 32, 64, 64, 'M', 128, 128, 'M', 196, 196, 'M', 256, 256]  # big model
    # model = crnn.get_crnn(config,cfg=cfg)
    model = myNet_ocr(num_classes=len(plate_chr), cfg=cfg)
    # model = build_lprnet(num_classes=len(plate_chr))

    # get device
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(config.GPUID))
    else:
        device = torch.device("cpu:0")
    # 将模型转移到目标设备
    model = model.to(device)

    # define loss function
    criterion = torch.nn.CTCLoss()

    last_epoch = config.TRAIN.BEGIN_EPOCH
    optimizer = utils.get_optimizer(config, model)
    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )

    if config.TRAIN.FINETUNE.IS_FINETUNE:
        model_state_file = config.TRAIN.FINETUNE.FINETUNE_CHECKPOINIT
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        model.load_state_dict(checkpoint)
    elif config.TRAIN.RESUME.IS_RESUME:
        model_state_file = config.TRAIN.RESUME.FILE
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['state_dict'])
            last_epoch = checkpoint['epoch']
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            model.load_state_dict(checkpoint)

    model_info(model)
    train_dataset = get_dataset(config)(config, input_w=config.WIDTH, input_h=config.HEIGHT, is_train=True)
    # DataLoader是PyTorch中用于数据加载的工具，它将数据集包装成可迭代的对象，方便按批次获取数据。
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,  # 是否打乱数据（True）
        num_workers=config.WORKERS,  # 数据加载进程数（如4）
        pin_memory=config.PIN_MEMORY,  # 启用快速内存拷贝（True）
    )

    val_dataset = get_dataset(config)(config, input_w=config.WIDTH, input_h=config.HEIGHT, is_train=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    best_acc = 0.5
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        function.train(config, train_loader, train_dataset, converter, model,
                       criterion, optimizer, device, epoch, writer_dict, output_dict)
        lr_scheduler.step()
        acc = function.validate(config, val_loader, val_dataset, converter,
                                model, criterion, device, epoch, writer_dict, output_dict)

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        print("is best:", is_best)
        print("best acc is:", best_acc)
        # save checkpoint
        torch.save(
            {
                "cfg": cfg,
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                # "optimizer": optimizer.state_dict(),
                # "lr_scheduler": lr_scheduler.state_dict(),
                "best_acc": best_acc,
            }, os.path.join(output_dict['chs_dir'], "checkpoint_{}_acc_{:.4f}.pth".format(epoch, acc))
        )

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()

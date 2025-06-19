import torch
from torch import nn
import lib.utils.utils as utils
from rec_alphabets import plate_chr
from rec_plateNet import myNet_ocr

model = myNet_ocr(num_classes=len(plate_chr), cfg=cfg)
converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
criterion = torch.nn.CTCLoss()
text, length = converter.encode(labels)

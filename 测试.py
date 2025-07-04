import torch
import lib.utils.utils as utils
from plate_recognition.rec_alphabets import plate_chr
from plate_recognition.rec_myNet import myNet

model = myNet(num_classes=len(plate_chr), cfg=cfg)
converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
criterion = torch.nn.CTCLoss()
text, length = converter.encode(labels)

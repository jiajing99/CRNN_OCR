import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
import lib.models.crnn as crnn
from lib.dataset import get_dataset
from lib.utils.utils import model_info
import lib.config.alphabets as alphabets
import lib.utils.utils as utils
import yaml

def testjpgCompress(model, device, val_dataset, val_loader, config):
    from lib.core.function import AverageMeter
    correct = 0
    losses = AverageMeter()

    criterion = torch.nn.CTCLoss()
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    optimizer = utils.get_optimizer(config, model)

    # Load the pretrained model
    model_state_file = 'output/OWN/crnn/2021-04-30-19-55/checkpoints/checkpoint_49_acc_0.9981.pth'
    if model_state_file == '':
        print(" => no checkpoint found")
    checkpoint = torch.load(model_state_file, map_location='cpu')
    if 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    model.load_state_dict(checkpoint)

    model.train()

    # Loop over all examples in test set
    for data, target in val_loader:
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        labels = utils.get_batch_label(val_dataset, target)

        # inference
        preds = model(data).cpu()

        # compute loss
        batch_size = data.size(0)
        text, length = converter.encode(labels)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        loss = criterion(preds, text, preds_size, length)

        losses.update(loss.item(), data.size(0))

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, labels):
            if pred == target:
                correct += 1

    num_test_sample = config.TEST.NUM_TEST_BATCH * config.TEST.BATCH_SIZE_PER_GPU
    if num_test_sample > len(val_dataset):
        num_test_sample = len(val_dataset)

    # Calculate final accuracy for this epsilon
    accuracy = correct / float(num_test_sample)
    print("Test loss: {:.4f}\tTest Accuracy = {} / {} = {:.4f}".format(losses.avg, correct, num_test_sample, accuracy))


def main():
    # load config
    with open('lib/config/OWN_config.yaml', 'r') as f:
        config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    # cudnn
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # construct face related neural networks
    model = crnn.get_crnn(config)

    # get device
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(config.GPUID))
    else:
        device = torch.device("cpu:0")

    model = model.to(device)
    model_info(model)

    val_dataset = get_dataset(config)(config, is_train=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    testjpgCompress(model, device, val_dataset, val_loader, config)


if __name__ == '__main__':
    main()

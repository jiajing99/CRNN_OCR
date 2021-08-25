import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
import lib.models.crnn as crnn
from lib.dataset import get_dataset
from lib.utils.utils import model_info
import lib.config.alphabets as alphabets
import lib.utils.utils as utils
import matplotlib
matplotlib.use('Agg')
import yaml
import argparse
import cv2

epsilons = [0, .05, .1, .15, .2, .25, .3]

class FGSM(nn.Module):  # η = εsign (∇xJ(θ, x, y))
    def __init__(self, model, labels, converter, criterion, optimizer):
        super().__init__()
        self.model = model
        self.labels = labels
        self.converter = converter
        self.criterion = criterion
        self.optimizer = optimizer

    def fgsm_attack(self, x, epsilon):
        # Set requires_grad attribute of tensor. Important for Attack
        x.requires_grad = True

        # inference
        preds = self.model(x).cpu()

        # compute loss
        batch_size = x.size(0)
        text, length = self.converter.encode(self.labels)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        loss = self.criterion(preds, text, preds_size, length)

        # Zero all existing gradients
        self.optimizer.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = x.grad.data

        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()

        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = x + epsilon * sign_data_grad

        # Adding clipping to maintain [0,1] range
        # perturbed_image = torch.clamp(perturbed_image, 0, 1)

        # Return the perturbed image
        return perturbed_image

def save(adv_x, label):
    std = 0.193
    mean = 0.588
    # print('adv_x',adv_x.shape)
    img = cv2.cvtColor(adv_x.data.cpu().numpy().transpose(1, 2, 0), cv2.COLOR_GRAY2RGB)
    cv2.imwrite("adv.jpg", (img * std + mean) * 255)
    
    with open('adv.txt', 'w', encoding='utf-8') as f:
        f.write(label)
    

def testFGSM(model, device, val_dataset, val_loader, config, epsilon, checkpointPath):
    from lib.core.function import AverageMeter
    correct = 0
    losses = AverageMeter()

    criterion = torch.nn.CTCLoss()
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    optimizer = utils.get_optimizer(config, model)

    # Load the pretrained model
    model_state_file = checkpointPath
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

        # Call FGSM Attack
        perturbed_data = FGSM(model, labels, converter, criterion, optimizer).fgsm_attack(data, epsilon)

        # Re-classify the perturbed image
        preds = model(perturbed_data).cpu()

        # compute loss
        batch_size = perturbed_data.size(0)
        text, length = converter.encode(labels)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        test_loss = criterion(preds, text, preds_size, length)

        losses.update(test_loss.item(), perturbed_data.size(0))

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for i, datas in enumerate(zip(sim_preds, labels)):
            pred, target = datas
            if pred == target:
                correct += 1
            save(perturbed_data[i], pred)

    num_test_sample = config.TEST.NUM_TEST_BATCH * config.TEST.BATCH_SIZE_PER_GPU
    if num_test_sample > len(val_dataset):
        num_test_sample = len(val_dataset)

    # Calculate final accuracy for this epsilon
    accuracy = correct / float(num_test_sample)
    print("Epsilon: {}\tTest loss: {:.4f}\tTest Accuracy = {} / {} = {:.4f}".format(epsilon, losses.avg, correct, num_test_sample, accuracy))

def parse_arg():
    parser = argparse.ArgumentParser(description="fgsm attack")
    parser.add_argument('--checkpoint', type=str, default='output/OWN/crnn/2021-02-23-16-02/checkpoints/checkpoint_19_acc_0.9982.pth', help='the path to your checkpoints')
    args = parser.parse_args()
    return args

def main():
    args = parse_arg()
    checkpointPath = args.checkpoint

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
    
    eps = 0.2
    testFGSM(model, device, val_dataset, val_loader, config, eps, checkpointPath)

# CUDA_VISIBLE_DEVICES=5
# python fgsmAttack1.py --checkpoint output/OWN/crnn/2021-02-23-16-02/checkpoints/checkpoint_19_acc_0.9982.pth
# python fgsmAttack1.py --checkpoint output/OWN/crnn/2021-04-14-14-29/checkpoints/checkpoint_49_acc_0.9945.pth
if __name__ == '__main__':
    main()

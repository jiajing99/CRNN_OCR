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

epsilons = [0, .05, .1, .15, .2, .25, .3]

class PGD(nn.Module):  # Xt+1 = Πx+S (Xt + αsign (∇xL(θ, x, y)))
    def __init__(self, model, device, labels, converter, criterion, optimizer):
        super().__init__()
        self.model = model
        self.device = device
        self.labels = labels
        self.converter = converter
        self.criterion = criterion
        self.optimizer = optimizer

        self.iter_eps = 0.01
        self.nb_iter = 40
        self.rand_init = False

    def single_step_attack(self, x):
        adv_x = x
        adv_x = Variable(adv_x)
        adv_x.requires_grad = True

        # inference
        preds = self.model(adv_x).cpu()
        preds = preds.to(self.device)

        # compute loss
        batch_size = adv_x.size(0)
        text, length = self.converter.encode(self.labels)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        loss = self.criterion(preds, text, preds_size, length)

        # Zero all existing gradients
        self.optimizer.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = adv_x.grad.data

        sign_data_grad = data_grad.sign()
        adv_x = x + self.iter_eps * sign_data_grad
        return adv_x

    def pgd_attack(self, x, epsilon):
        if self.rand_init:
            x_tmp = x + torch.Tensor(np.random.uniform(-self.eps, self.eps, x.shape)).type_as(x).cuda()
        else:
            x_tmp = x

        x = x.cpu().detach().numpy()
        adv_x = x_tmp

        for i in range(self.nb_iter):
            next_adv_x = self.single_step_attack(adv_x)
            next_adv_x = next_adv_x.cpu().detach().numpy()
            next_adv_x = np.clip(next_adv_x, x - epsilon, x + epsilon)
            adv_x = torch.from_numpy(next_adv_x).cuda()

        return adv_x

def usePGD(model, device, labels, converter, criterion, optimizer, data, epsilon):
    perturbed_data = PGD(model, device, labels, converter, criterion, optimizer).pgd_attack(data, epsilon)
    return perturbed_data

def testPGD(model, device, val_dataset, val_loader, config, epsilon):
    from lib.core.function import AverageMeter
    correct = 0
    losses = AverageMeter()

    criterion = torch.nn.CTCLoss()
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    optimizer = utils.get_optimizer(config, model)

    # Load the pretrained model
    model_state_file = 'output/OWN/crnn/2020-12-11-16-41/checkpoints/checkpoint_2_acc_0.9937.pth'
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

        # Call PGD Attack
        perturbed_data = PGD(model, device, labels, converter, criterion, optimizer).pgd_attack(data, epsilon)

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
        for pred, target in zip(sim_preds, labels):
            if pred == target:
                correct += 1

    num_test_sample = config.TEST.NUM_TEST_BATCH * config.TEST.BATCH_SIZE_PER_GPU
    if num_test_sample > len(val_dataset):
        num_test_sample = len(val_dataset)

    # Calculate final accuracy for this epsilon
    accuracy = correct / float(num_test_sample)
    print("Epsilon: {}\tTest loss: {:.4f}\tTest Accuracy = {} / {} = {:.4f}".format(epsilon, losses.avg, correct, num_test_sample, accuracy))


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

    for eps in epsilons:
        testPGD(model, device, val_dataset, val_loader, config, eps)


if __name__ == '__main__':
    main()

import torch.optim as optim
from net.network import FQ_LISCS
from data.datasets import get_loader
from utils import *
import matplotlib.pyplot as plt
import os
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from datetime import datetime
import torch.nn as nn
import argparse
from loss.distortion import *
import time
from quantization_utils import QuantLinear, QuantAct, QuantConv2d, IntLayerNorm, IntSoftmax, IntGELU, QuantMatMul
from utils import *
from model_utils import *

parser = argparse.ArgumentParser(description='FQ_LISCS')
parser.add_argument('--training', action='store_true',
                    help='training or testing')
parser.add_argument('--trainset', type=str, default='CIFAR10',
                    choices=['CIFAR10', 'DIV2K'],
                    help='train dataset name')
parser.add_argument('--testset', type=str, default='kodak',
                    choices=['kodak', 'CLIC21'],
                    help='specify the testset for HR models')
parser.add_argument('--distortion-metric', type=str, default='MSE',
                    choices=['MSE', 'MS-SSIM'],
                    help='evaluation metrics')

parser.add_argument('--channel-type', type=str, default='awgn',
                    choices=['awgn', 'rayleigh'],
                    help='wireless channel model, awgn or rayleigh')
parser.add_argument('--C', type=int, default=96,
                    help='bottleneck dimension')
parser.add_argument('--multiple-snr', type=str, default='1,4,7,10,13',
                    help='random or fixed snr')
args = parser.parse_args()


class config():
    seed = 1024
    pass_channel = True
    CUDA = True
    device = torch.device("cuda:0")
    norm = False
    # logger
    print_step = 100
    plot_step = 10000
    filename = "TEST"
    workdir = './history/{}'.format(filename)
    log = workdir + '/Log_{}.log'.format(filename)
    samples = workdir + '/samples'
    models = workdir + '/models'
    results_folder=workdir+'/picture'
    logger = None

    # training details
    normalize = False
    learning_rate = 0.00001
    tot_epoch = 10000000

    if args.trainset == 'CIFAR10':
        save_model_freq = 1
        image_dims = (3, 32, 32)
        train_data_dir = "/home/myserver/FQ_LISCS/media/Dataset/CIFAR10/"
        test_data_dir = "/home/myserver/FQ_LISCS/media/Dataset/CIFAR10/"
        batch_size = 128  # 128
        downsample = 2
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8], C=args.C,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=IntLayerNorm, patch_norm=True,
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4], C=args.C,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=IntLayerNorm, patch_norm=True,
        )
    elif args.trainset == 'DIV2K':
        save_model_freq = 1
        image_dims = (3, 256, 256)
        train_data_dir = ["/home/myserver/FQ_LISCS/media/Dataset/HR_Image_dataset/DIV2K_train_HR"]
        if args.testset == 'kodak':
            test_data_dir = ["/home/myserver/FQ_LISCS/media/Dataset/kodak_test/"]
        elif args.testset == 'CLIC21':
            test_data_dir = ["/home/myserver/FQ_LISCS/media/Dataset/CLIC21/"]
        batch_size = 4
        downsample = 4
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10],
            C=args.C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=IntLayerNorm, patch_norm=True,
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4],
            C=args.C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=IntLayerNorm, patch_norm=True,
        )


if args.trainset == 'CIFAR10':
    CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
else:
    CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()


def load_weights(model_path):
    pretrained = torch.load(model_path)
    new_pretrained = {}
    for key in pretrained:
        new_key = key
        for i in range(7):
            new_key = (new_key
                       .replace(f'encoder.bm_list.{i}.fc.0.weight', f'encoder.bm_list.{i}.QLinear1.weight')
                       .replace(f'encoder.bm_list.{i}.fc.0.bias', f'encoder.bm_list.{i}.QLinear1.bias')
                       .replace(f'encoder.bm_list.{i}.fc.2.weight', f'encoder.bm_list.{i}.QLinear2.weight')
                       .replace(f'encoder.bm_list.{i}.fc.2.bias', f'encoder.bm_list.{i}.QLinear2.bias')
                       .replace(f'encoder.bm_list.{i}.fc.4.weight', f'encoder.bm_list.{i}.QLinear3.weight')
                       .replace(f'encoder.bm_list.{i}.fc.4.bias', f'encoder.bm_list.{i}.QLinear3.bias')
                       .replace(f'decoder.bm_list.{i}.fc.0.weight', f'decoder.bm_list.{i}.QLinear1.weight')
                       .replace(f'decoder.bm_list.{i}.fc.0.bias', f'decoder.bm_list.{i}.QLinear1.bias')
                       .replace(f'decoder.bm_list.{i}.fc.2.weight', f'decoder.bm_list.{i}.QLinear2.weight')
                       .replace(f'decoder.bm_list.{i}.fc.2.bias', f'decoder.bm_list.{i}.QLinear2.bias')
                       .replace(f'decoder.bm_list.{i}.fc.4.weight', f'decoder.bm_list.{i}.QLinear3.weight')
                       .replace(f'decoder.bm_list.{i}.fc.4.bias', f'decoder.bm_list.{i}.QLinear3.bias'))
        #This model is based on the WITT pre-training model for quantitative perception training, and there are differences 
        # in the OPERATION naming of the model to improve readability, which is used here for conversion.
        new_pretrained[new_key] = pretrained[key]

    net.load_state_dict(new_pretrained, strict=False)

    A = net.state_dict()
    del pretrained

def train_one_epoch(args):
    unfreeze_model(net)
    random_int = random.randrange(0, 2000)
    torch.manual_seed(random_int)
    net.train()
    elapsed, losses, psnrs, msssims, cbrs, snrs = [AverageMeter() for _ in range(6)]
    metrics = [elapsed, losses, psnrs, msssims, cbrs, snrs]
    global global_step
    if args.trainset == 'CIFAR10':
        for batch_idx, (input, label) in enumerate(train_loader):
            start_time = time.time()
            global_step += 1
            input = input.cuda()
            recon_image, CBR, SNR, mse, loss_G = net(input)
            loss = loss_G
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            cbrs.update(CBR)
            snrs.update(SNR)
            if mse.item() > 0:
                psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                psnrs.update(psnr.item())
                msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                msssims.update(msssim)
            else:
                psnrs.update(100)
                msssims.update(100)

            if (global_step % config.print_step) == 0:
                process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
                log = (' | '.join([
                    f'Epoch {epoch}',
                    f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                    f'Time {elapsed.val:.3f}',
                    f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                    f'Lr {cur_lr}',
                ]))
                logger.info(log)
                for i in metrics:
                    i.clear()
    else:
        for batch_idx, input in enumerate(train_loader):
            start_time = time.time()
            global_step += 1
            input = input.cuda()
            recon_image, CBR, SNR, mse, loss_G = net(input)
            loss = loss_G
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            cbrs.update(CBR)
            snrs.update(SNR)
            if mse.item() > 0:
                psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                psnrs.update(psnr.item())
                msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                msssims.update(msssim)

            else:
                psnrs.update(100)
                msssims.update(100)

            if (global_step % config.print_step) == 0:
                process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
                log = (' | '.join([
                    f'Epoch {epoch}',
                    f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                    f'Time {elapsed.val:.3f}',
                    f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                    f'Lr {cur_lr}',
                ]))
                logger.info(log)
                for i in metrics:
                    i.clear()
    for i in metrics:
        i.clear()


def test():
    torch.manual_seed(1024)
    config.isTrain = False
    net.eval()
    freeze_model(net)
    elapsed, psnrs, msssims, snrs, cbrs = [AverageMeter() for _ in range(5)]
    metrics = [elapsed, psnrs, msssims, snrs, cbrs]
    multiple_snr = args.multiple_snr.split(",")
    for i in range(len(multiple_snr)):
        multiple_snr[i] = int(multiple_snr[i])
    results_snr = np.zeros(len(multiple_snr))
    results_cbr = np.zeros(len(multiple_snr))
    results_psnr = np.zeros(len(multiple_snr))
    results_msssim = np.zeros(len(multiple_snr))
    for i, SNR in enumerate(multiple_snr):
        with torch.no_grad():
            if args.trainset == 'CIFAR10':
                for batch_idx, (input, label) in enumerate(test_loader):
                    start_time = time.time()
                    input = input.cuda()
                    recon_image, CBR, SNR, mse, loss_G = net(input, SNR)
                    elapsed.update(time.time() - start_time)
                    cbrs.update(CBR)
                    snrs.update(SNR)
                    if mse.item() > 0:
                        psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                        psnrs.update(psnr.item())
                        msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                        msssims.update(msssim)
                    else:
                        psnrs.update(100)
                        msssims.update(100)

                    log = (' | '.join([
                        f'Time {elapsed.val:.3f}',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f}',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'Lr {cur_lr}',
                    ]))
                    logger.info(log)
            else:
                for batch_idx, input in enumerate(test_loader):
                    start_time = time.time()
                    input = input.cuda()
                    recon_image, CBR, SNR, mse, loss_G = net(input, SNR)
                    elapsed.update(time.time() - start_time)
                    cbrs.update(CBR)
                    snrs.update(SNR)
                    if mse.item() > 0:
                        psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                        psnrs.update(psnr.item())
                        msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                        msssims.update(msssim)
                    else:
                        psnrs.update(100)
                        msssims.update(100)

                    log = (' | '.join([
                        f'Time {elapsed.val:.3f}',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f}',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'Lr {cur_lr}',
                    ]))
                    logger.info(log)
                    torch.cuda.empty_cache()
        results_snr[i] = snrs.avg
        results_cbr[i] = cbrs.avg
        results_psnr[i] = psnrs.avg
        results_msssim[i] = msssims.avg
        for t in metrics:
            t.clear()
    if len(results_psnr) > 0:
        first_psnrs.append(results_psnr[0])
        last_psnrs.append(results_psnr[-1])
        plt.clf()  
        plt.plot(first_psnrs, label='First PSNR', marker='o')
        plt.plot(last_psnrs, label='Last PSNR', marker='x')
        plt.title("First and Last PSNR Values Over Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("PSNR")
        plt.legend()
        plt.savefig(f"{config.results_folder}/PSNR_first_last_changes.png")
        plt.close()
    print("SNR: {}".format(results_snr.tolist()))
    print("CBR: {}".format(results_cbr.tolist()))
    print("PSNR: {}".format(results_psnr.tolist()))
    print("MS-SSIM: {}".format(results_msssim.tolist()))
    print("Finish Test!")

 
if __name__ == '__main__':
    seed_torch()
    logger = logger_configuration(config, save_log=True)
    logger.info(config.__dict__)
    torch.manual_seed(seed=config.seed)
    net = FQ_LISCS(args, config)
    
    model_path = "/home/myserver/FQ_LISCS/history/div2k_awgn_snr1471013_C96/models/div2k_awgn_snr1471013_C96.model"  
    load_weights(model_path)
    net = net.cuda()
    model_params = [{'params': net.parameters(), 'lr': 0.00001}]
    train_loader, test_loader = get_loader(args, config)
    cur_lr = config.learning_rate
    optimizer = optim.Adam(model_params, lr=cur_lr)
    global_step = 0
    steps_epoch = global_step // train_loader.__len__()
    first_psnrs = []
    last_psnrs = []
    if not os.path.exists(config.results_folder):
      os.makedirs(config.results_folder)
    if args.training:
        for epoch in range(steps_epoch, config.tot_epoch):
            train_one_epoch(args)
            if (epoch + 1) % config.save_model_freq == 0:

                save_model(net, save_path=config.models + '/{}_EP{}.model'.format(config.filename, epoch + 1))#-26

                test()
    else:
        test()
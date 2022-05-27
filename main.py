import argparse
import torch
from data_provider.data_factory import data_provider
from torch import optim
from module.model import TCN, MLP
from torch import nn
import time
import numpy as np
from utils.tool import adjust_learning_rate,metric,visual
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from utils.pyplot import plot_seq_feature, plot_heatmap_feature



parser = argparse.ArgumentParser(description='Sequence Modeling - The Adding Problem')
parser.add_argument('--is_training', type=int, default=1, help='status')

parser.add_argument('--device', type=int, default=0, help='gpu dvice')

#method choose
parser.add_argument('--use_wv',type=int,default=1,help='-1->only permute in 1d, 0->use repeat,1->use wv,2->use pseudo_wv')
parser.add_argument('--exp_name', type=str, default='baseline', help='Experiment name')
parser.add_argument('--model_type', type=str,choices=['TCN', 'MLP', 'swin_Transformer', 'swin_MLP'], \
    default='TCN', help='model type')

# data loader
parser.add_argument('--data', type=str, default='Spectrum', help='dataset type')
parser.add_argument('--data_path', type=str, default='data/psd.mat', help='data file')
parser.add_argument('--save_dir',type=str,default='/media/ps/passport1/time-series-forecasting/after_data')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', default=[24], help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--kernel',type=int,default=3,help='kernel size for conv layer')

# optimization
parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
parser.add_argument('--itr', type=int, default=3, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=5e-3, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type4', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multi gpus')

#checkpoint_path
parser.add_argument('--check_point',type=str,default='checkpoint',help='check point path')

#hiden layer
parser.add_argument('--hiden',type=int,default=128,help='hiden channel')
parser.add_argument('--layer',type=int,default=7,help='layer of block')

parser.add_argument('--threshold',type=float,default=-106.5, help='threshold of occupancy')
parser.add_argument('--use_BCELoss',action='store_true', help='if use BCE loss')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

print('Args in experiment:')
print(args)


if args.model_type == 'TCN':
    num_channels = [args.hiden] * args.layer
    model = TCN(args.enc_in,num_channels,args.kernel,args.dropout,args.seq_len,args.pred_len)
elif args.model_type == 'MLP':
    model = MLP(args.seq_len,args.pred_len, middle=args.hiden)
print(model)

device = "cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu"
model.to(device)  
expname = '{}/{}/{}_in{}out{}_dropout{}_lr{}_bs{}_epoch{}' \
        .format(args.data, args.model_type, \
            args.exp_name, args.seq_len, args.pred_len, args.dropout, \
                args.learning_rate, args.batch_size, args.train_epochs)
print(expname)



def vali(vali_data, vali_loader, criterion, epoch, writer, flag='vali'):
    total_loss = []
    total_ocpy_acc = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, threshold) in enumerate(vali_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()

            batch_y = batch_y[:, -args.pred_len:, :].to(device)
            outputs = model(batch_x)
            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()
            loss = criterion(pred, true)
            total_loss.append(loss)

            threshold = threshold.float().unsqueeze(1).repeat(1, outputs.shape[1], 1).to(device)
            pred_ocpy = (outputs > threshold)
            true_ocpy = (batch_y > threshold)
            ocpy_acc = torch.mean((~(pred_ocpy^true_ocpy)).float()).detach().cpu().numpy()
            total_ocpy_acc.append(ocpy_acc)

            if i == 0:
                fig = plot_seq_feature(outputs, batch_y, batch_x, threshold, flag)
                writer.add_figure("figure_{}".format(flag), fig, global_step=epoch)

    total_loss = np.average(total_loss)
    total_ocpy_acc = np.average(total_ocpy_acc)
    model.train()
    return total_loss, total_ocpy_acc

def train():
    train_set, train_loader = data_provider(args, "train")
    vali_data, vali_loader = data_provider(args,flag='val')
    test_data, test_loader = data_provider(args,flag='test')
    
    optimizer = optim.SGD(model.parameters(),lr=args.learning_rate,momentum=0.9)
    criterion = nn.MSELoss()
    criterion_BCE = nn.BCEWithLogitsLoss() if args.use_BCELoss else None

    train_steps = len(train_loader)
    writer = SummaryWriter('event/{}'.format(expname))
    # log args setting
    argsDict = args.__dict__
    folder_path = os.path.join('./test_results', expname)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(folder_path + '/setting.txt','w') as f:
        f.writelines('------------------start--------------------\n')
        for eachArg,value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------end--------------------')

    # start train
    best_loss = 0
    for epoch in range(args.train_epochs):
        train_loss = []
        train_ocpy_acc = []
        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y, threshold) in enumerate(train_loader):
            optimizer.zero_grad()

            # to cuda
            batch_x = batch_x.float().to(device) # (B,L,C)
            batch_y = batch_y.float().to(device) # (B,L,C)
            outputs = model(batch_x) # (B,L,C)
            batch_y = batch_y[:, -args.pred_len:, :].to(device)

            threshold = threshold.float().unsqueeze(1).repeat(1, outputs.shape[1], 1).to(device)
            pred_ocpy = (outputs > threshold)
            true_ocpy = (batch_y > threshold)
            ocpy_acc = torch.mean((~(pred_ocpy^true_ocpy)).float()).detach().cpu().numpy()
            train_ocpy_acc.append(ocpy_acc)
            
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())
            loss_BCE = 0
            if args.use_BCELoss:
                loss_BCE = criterion_BCE(outputs-threshold, true_ocpy.float())
                loss += loss_BCE
                loss_BCE = loss_BCE.item()
            loss.backward()
            optimizer.step()
            

            if (i+1) % (train_steps//5) == 0:
                print("\titers: {0}, epoch: {1} | loss_MSE: {2:.7f} loss_BCE: {3:.7f} ocpy_acc: {4:.4f}".format( \
                    i + 1, epoch + 1, loss.item()-loss_BCE, loss_BCE, ocpy_acc))


        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

        train_loss = np.average(train_loss)
        train_ocpy_acc = np.average(train_ocpy_acc)
        vali_loss, vali_ocpy_acc = vali(vali_data, vali_loader, criterion, epoch, writer, 'vali')
        test_loss, test_ocpy_acc = vali(test_data, test_loader, criterion, epoch, writer, 'test')

        print("Epoch: {0} | Train ocpy_acc: {1:.4f} Vali ocpy_acc: {2:.4f} Test ocpy_acc: {3:.4f} \n\
            Train Loss: {4:.7f} Vali Loss: {5:.7f} Test Loss: {6:.7f}".format(
            epoch + 1, train_ocpy_acc, vali_ocpy_acc, test_ocpy_acc, train_loss, vali_loss, test_loss))

        fig = plot_seq_feature(outputs, batch_y, batch_x, threshold)
        writer.add_figure("figure_train", fig, global_step=epoch)
        writer.add_scalar('train_ocpy_acc', train_ocpy_acc, global_step=epoch)
        writer.add_scalar('vali_ocpy_acc', vali_ocpy_acc, global_step=epoch)
        writer.add_scalar('test_ocpy_acc', test_ocpy_acc, global_step=epoch)
        writer.add_scalar('train_loss', train_loss, global_step=epoch)
        writer.add_scalar('vali_loss', vali_loss, global_step=epoch)
        writer.add_scalar('test_loss', test_loss, global_step=epoch)
        
        ckpt_path = os.path.join(args.check_point, expname)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        if best_loss == 0:
            best_loss = vali_loss
            torch.save(model.state_dict(), os.path.join(ckpt_path, 'valid_best_checkpoint.pth'))
        else:
            if vali_loss < best_loss:
                torch.save(model.state_dict(), os.path.join(ckpt_path, 'valid_best_checkpoint.pth'))

        torch.save(model.state_dict(), os.path.join(ckpt_path, 'final_checkpoint.pth'))
        
        adjust_learning_rate(optimizer, epoch + 1, args)

    return


def test(setting='setting',test=1):
    test_data, test_loader = data_provider(args,flag='test')
    if test:
        print('loading model')
        model.load_state_dict(torch.load(os.path.join(args.check_point, expname, 'valid_best_checkpoint.pth')))

    preds = []
    trues = []
    folder_path = os.path.join('./test_results', expname, 'visual')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, threshold) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            outputs = model(batch_x)
            batch_y = batch_y[:, -args.pred_len:, :].to(device)
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
            true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

            preds.append(pred)
            trues.append(true)
            if i % 20 == 0:
                input = batch_x.detach().cpu().numpy()
                gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

    preds = np.array(preds)
    trues = np.array(trues)
    print('test shape:', preds.shape, trues.shape)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print('test shape:', preds.shape, trues.shape)

    # result save
    folder_path = os.path.join('./test_results', expname, 'metrics')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print('mse:{}, mae:{}'.format(mse, mae))
    f = open(os.path.join('./test_results', expname, 'test_result.txt'), 'a')
    f.write(setting + "  \n")
    f.write('mse:{}, mae:{}'.format(mse, mae))
    f.write('\n')
    f.write('\n')
    f.close()

    np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
    np.save(folder_path + 'pred.npy', preds)
    np.save(folder_path + 'true.npy', trues)

    return

def pred():
    test_data, test_loader = data_provider(args,flag='test')
    print('-------------------loading model-------------------')
    model.load_state_dict(torch.load(os.path.join(args.check_point, expname, 'valid_best_checkpoint.pth')))

    preds = []
    trues = []
    folder_path = os.path.join('./test_results', expname)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, threshold) in enumerate(test_loader):
            # speed test     
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            if i == 1:
                with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
                    outputs = model(batch_x)
                print(prof.table())
                prof.export_chrome_trace(folder_path + '/profile.json')
                break

            outputs = model(batch_x)
            batch_y = batch_y[:, -args.pred_len:, :].to(device)
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
            true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

            preds.append(pred)
            trues.append(true)

            fig = plot_heatmap_feature(outputs, batch_y, batch_x)
            plt.savefig('heatmap_{}'.format(i))

            
    preds = np.array(preds)
    trues = np.array(trues)
    print('test shape:', preds.shape, trues.shape)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print('test shape:', preds.shape, trues.shape)

    # result save
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print('mse:{}, mae:{}'.format(mse, mae))

    # log metric and args setting
    argsDict = args.__dict__
    with open(folder_path + '/pred_result.txt','w') as f:
        f.write("metrics  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.writelines('Args  \n')
        for eachArg,value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')

    return



#main
train()
# test()
# pred()





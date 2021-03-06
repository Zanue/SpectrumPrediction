import argparse
import torch
from data_provider.data_factory import data_provider
from torch import optim
# from module.model import TCN, MLP
from module.tcn.model import TCN
from module.mlp.mlp import MLP
from module.informer.model import Informer
from module.mlpmixer.mlpmixer import MLPMixer
from module.rnn.rnn import RNNet
from module.lstm.lstm import LSTMnet
from torch import nn
import time
import numpy as np
from utils.tool import adjust_learning_rate,metric,visual
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from utils.pyplot import plot_seq_feature, plot_heatmap_feature
from thop import profile
from thop import clever_format


parser = argparse.ArgumentParser(description='Sequence Modeling - The Adding Problem')
parser.add_argument('--is_training', type=int, default=1, help='status')

parser.add_argument('--device', type=int, default=0, help='gpu dvice')

#method choose
parser.add_argument('--exp_name', type=str, default='baseline', help='Experiment name')
parser.add_argument('--model_type', type=str,choices=['TCN', 'MLP', 'informer', 'MLPMixer', 'rnn', 'lstm'], \
    default='TCN', help='model type')

# data loader
parser.add_argument('--data', type=str, default='Spectrum', help='dataset type')
parser.add_argument('--data_path', type=str, default='data/psd.mat', help='data file')
parser.add_argument('--save_dir',type=str,default='/media/ps/passport1/time-series-forecasting/after_data')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# model define
parser.add_argument('--enc_in', type=int, default=2048, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=2048, help='decoder input size')
parser.add_argument('--c_out', type=int, default=2048, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', default=[24], help='window size of moving average')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
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
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)

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
parser.add_argument('--hiden',type=int,default=2048,help='hiden channel')
parser.add_argument('--layer',type=int,default=7,help='layer of block')
parser.add_argument('--rnn_layers',type=int,default=1,help='rnn layers num')

parser.add_argument('--threshold',type=float,default=-111.5, help='threshold of occupancy')
parser.add_argument('--resume',type=str,default=None, help='resume path')


args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

print('Args in experiment:')
print(args)

device = "cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu"

if args.model_type == 'TCN':
    num_channels = [args.hiden] * args.layer
    model = TCN(args.enc_in,num_channels,args.kernel,args.dropout,args.seq_len,args.pred_len)
    expname = '{}/{}/{}_in{}out{}_hiden{}_layer{}_lr{}_bs{}_epoch{}' \
        .format(args.data, args.model_type, \
            args.exp_name, args.seq_len, args.pred_len, args.hiden, args.layer, \
                args.learning_rate, args.batch_size, args.train_epochs)
elif args.model_type == 'MLP':
    model = MLP(args.seq_len,args.pred_len, middle=args.hiden)
    expname = '{}/{}/{}_in{}out{}_middle{}_lr{}_bs{}_epoch{}' \
        .format(args.data, args.model_type, \
            args.exp_name, args.seq_len, args.pred_len, args.hiden, \
                args.learning_rate, args.batch_size, args.train_epochs)
elif args.model_type == 'informer':
    model = Informer(args.enc_in, args.dec_in,args.c_out, args.seq_len, args.label_len,args.pred_len, \
        args.factor, args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.dropout, args.attn, \
        args.embed, args.freq, args.activation, args.output_attention, args.distil, args.mix, device)
    expname = '{}/{}/{}_in{}out{}_en{}de{}_dmodel{}dff{}_nhead{}_lr{}_bs{}_epoch{}' \
        .format(args.data, args.model_type, \
            args.exp_name, args.seq_len, args.pred_len, args.e_layers, args.d_layers, \
                args.d_model, args.d_layers, args.n_heads, \
                args.learning_rate, args.batch_size, args.train_epochs)
elif args.model_type == 'MLPMixer':
    model = MLPMixer(args.seq_len, args.pred_len, num_channels=args.enc_in, num_blocks=args.layer)
    expname = '{}/{}/{}_in{}out{}_lr{}_bs{}_blocks{}_epoch{}' \
        .format(args.data, args.model_type, \
            args.exp_name, args.seq_len, args.pred_len, \
                args.learning_rate, args.batch_size, args.layer, args.train_epochs)
elif args.model_type == 'rnn':
    model=RNNet(num_layers=args.rnn_layers,seq_len=args.seq_len,pred_len=args.pred_len)
    expname = '{}/{}/{}_in{}out{}_layer{}_lr{}_bs{}_epoch{}' \
        .format(args.data, args.model_type, \
            args.exp_name, args.seq_len, args.pred_len, args.rnn_layers, \
                args.learning_rate, args.batch_size, args.train_epochs)
elif args.model_type == 'lstm':
    model=LSTMnet(num_layers=args.rnn_layers,seq_len=args.seq_len,pred_len=args.pred_len)
    expname = '{}/{}/{}_in{}out{}_layer{}_lr{}_bs{}_epoch{}' \
        .format(args.data, args.model_type, \
            args.exp_name, args.seq_len, args.pred_len, args.rnn_layers, \
                args.learning_rate, args.batch_size, args.train_epochs)


print(model)
if args.resume is not None:
    print('loading model')
    model.load_state_dict(torch.load(args.resume))
model.to(device)
print(expname)



def vali(vali_data, vali_loader, criterion, epoch, writer, flag='vali'):
    total_loss = []
    total_ocpy_acc = []
    total_false_alarm = []
    total_missing_alarm = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, threshold) in enumerate(vali_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()
            batch_y = batch_y[:, -args.pred_len:, :].to(device)
            
            if args.model_type == 'informer':
                dec_inp = torch.zeros([batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).float().to(device)
                dec_inp = torch.cat([batch_y[:,:args.label_len,:], dec_inp], dim=1).float()
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.model_type == 'informer':
                        outputs = model(batch_x, dec_inp) # (B,L,C)
                    else:
                        outputs = model(batch_x) # (B,L,C)
            else:
                if args.model_type == 'informer':
                    dec_inp = torch.zeros([batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).float().to(device)
                    dec_inp = torch.cat([batch_y[:,:args.label_len,:], dec_inp], dim=1).float()
                    outputs = model(batch_x, dec_inp) # (B,L,C)
                else:
                    outputs = model(batch_x) # (B,L,C)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()
            loss = criterion(pred, true)
            total_loss.append(loss)

            # threshold = threshold.float().unsqueeze(1).repeat(1, outputs.shape[1], 1).to(device)
            threshold = threshold.float().unsqueeze(1).unsqueeze(2).repeat(1, outputs.shape[1], outputs.shape[2]).to(device)
            pred_ocpy = (outputs > threshold)
            true_ocpy = (batch_y > threshold)
            ocpy_acc = torch.mean((~(pred_ocpy^true_ocpy)).float()).detach().cpu().numpy()
            total_ocpy_acc.append(ocpy_acc)
            false_alarm = torch.mean((pred_ocpy&(~true_ocpy)).float()).detach().cpu().numpy()
            total_false_alarm.append(false_alarm)
            missing_alarm = torch.mean(((~pred_ocpy)&true_ocpy).float()).detach().cpu().numpy()
            total_missing_alarm.append(missing_alarm)

            if i == 0:
                fig = plot_seq_feature(outputs, batch_y, batch_x, threshold, flag)
                writer.add_figure("figure_{}".format(flag), fig, global_step=epoch)

    total_loss = np.average(total_loss)
    total_ocpy_acc = np.average(total_ocpy_acc)
    total_false_alarm = np.average(total_false_alarm)
    total_missing_alarm = np.average(total_missing_alarm)
    model.train()
    return total_loss, total_ocpy_acc, total_false_alarm, total_missing_alarm


def train():
    train_set, train_loader = data_provider(args, "train")
    vali_data, vali_loader = data_provider(args,flag='val')
    test_data, test_loader = data_provider(args,flag='test')
    
    optimizer = optim.SGD(model.parameters(),lr=args.learning_rate,momentum=0.9)
    criterion = nn.MSELoss()

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

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # start train
    best_loss = 0
    for epoch in range(args.train_epochs):
        train_loss = []
        train_ocpy_acc = []
        train_false_alarm = []
        train_missing_alarm = []
        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y, threshold) in enumerate(train_loader):
            optimizer.zero_grad()

            # to cuda
            batch_x = batch_x.float().to(device) # (B,L,C)
            batch_y = batch_y.float().to(device) 
            batch_y = batch_y[:, -args.pred_len:, :].to(device) # (B,L,C)


            if args.model_type == 'informer':
                dec_inp = torch.zeros([batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).float().to(device)
                dec_inp = torch.cat([batch_y[:,:args.label_len,:], dec_inp], dim=1).float()
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.model_type == 'informer':
                        outputs = model(batch_x, dec_inp) # (B,L,C)
                    else:
                        outputs = model(batch_x) # (B,L,C)
                    loss = criterion(outputs, batch_y)
            else:
                if args.model_type == 'informer':
                    dec_inp = torch.zeros([batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).float().to(device)
                    dec_inp = torch.cat([batch_y[:,:args.label_len,:], dec_inp], dim=1).float()
                    outputs = model(batch_x, dec_inp) # (B,L,C)
                else:
                    outputs = model(batch_x) # (B,L,C)
                loss = criterion(outputs, batch_y)
            
            # threshold = threshold.float().unsqueeze(1).repeat(1, outputs.shape[1], 1).to(device)
            threshold = threshold.float().unsqueeze(1).unsqueeze(2).repeat(1, outputs.shape[1], outputs.shape[2]).to(device)
            pred_ocpy = (outputs > threshold)
            true_ocpy = (batch_y > threshold)
            ocpy_acc = torch.mean((~(pred_ocpy^true_ocpy)).float()).detach().cpu().numpy()
            train_ocpy_acc.append(ocpy_acc)
            false_alarm = torch.mean((pred_ocpy&(~true_ocpy)).float()).detach().cpu().numpy()
            train_false_alarm.append(false_alarm)
            missing_alarm = torch.mean(((~pred_ocpy)&true_ocpy).float()).detach().cpu().numpy()
            train_missing_alarm.append(missing_alarm)
            
            train_loss.append(loss.item())

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            if (i+1) % (train_steps//5) == 0:
                print("\titers: {0}, epoch: {1} | loss_MSE: {2:.7f}  ocpy_acc: {3:.4f} false_alarm: {4:.4f} missing_alarm: {5:.4f}".format( \
                    i + 1, epoch + 1, loss.item(), ocpy_acc, false_alarm, missing_alarm))


        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

        train_loss = np.average(train_loss)
        train_ocpy_acc = np.average(train_ocpy_acc)
        train_false_alarm = np.average(train_false_alarm)
        train_missing_alarm = np.average(train_missing_alarm)
        vali_loss, vali_ocpy_acc, vali_false_alarm, vali_missing_alarm = vali(vali_data, vali_loader, criterion, epoch, writer, 'vali')
        test_loss, test_ocpy_acc, test_false_alarm, test_missing_alarm = vali(test_data, test_loader, criterion, epoch, writer, 'test')

        print("Epoch: {0} | Train ocpy_acc: {1:.4f} Vali ocpy_acc: {2:.4f} Test ocpy_acc: {3:.4f} \n\
            Train false_alarm: {4:.4f} Vali false_alarm: {5:.4f} Test false_alarm: {6:.4f} \n\
            Train missing_alarm: {7:.4f} Vali missing_alarm: {8:.4f} Test missing_alarm: {9:.4f} \n\
            Train Loss: {10:.7f} Vali Loss: {11:.7f} Test Loss: {12:.7f}".format( \
            epoch + 1, train_ocpy_acc, vali_ocpy_acc, test_ocpy_acc, \
            train_false_alarm, vali_false_alarm, test_false_alarm, \
            train_missing_alarm, vali_missing_alarm, test_missing_alarm, \
            train_loss, vali_loss, test_loss))

        fig = plot_seq_feature(outputs, batch_y, batch_x, threshold)
        writer.add_figure("figure_train", fig, global_step=epoch)
        writer.add_scalar('train_ocpy_acc', train_ocpy_acc, global_step=epoch)
        writer.add_scalar('vali_ocpy_acc', vali_ocpy_acc, global_step=epoch)
        writer.add_scalar('test_ocpy_acc', test_ocpy_acc, global_step=epoch)
        writer.add_scalar('train_false_alarm', train_false_alarm, global_step=epoch)
        writer.add_scalar('vali_false_alarm', vali_false_alarm, global_step=epoch)
        writer.add_scalar('test_false_alarm', test_false_alarm, global_step=epoch)
        writer.add_scalar('train_missing_alarm', train_missing_alarm, global_step=epoch)
        writer.add_scalar('vali_missing_alarm', vali_missing_alarm, global_step=epoch)
        writer.add_scalar('test_missing_alarm', test_missing_alarm, global_step=epoch)
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
            batch_y = batch_y[:, -args.pred_len:, :].to(device)

            if args.model_type == 'informer':
                dec_inp = torch.zeros([batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).float().to(device)
                dec_inp = torch.cat([batch_y[:,:args.label_len,:], dec_inp], dim=1).float()
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.model_type == 'informer':
                        outputs = model(batch_x, dec_inp) # (B,L,C)
                    else:
                        outputs = model(batch_x) # (B,L,C)
            else:
                if args.model_type == 'informer':
                    dec_inp = torch.zeros([batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).float().to(device)
                    dec_inp = torch.cat([batch_y[:,:args.label_len,:], dec_inp], dim=1).float()
                    outputs = model(batch_x, dec_inp) # (B,L,C)
                else:
                    outputs = model(batch_x) # (B,L,C)
            
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
            batch_y = batch_y[:, -args.pred_len:, :].to(device)

            # latency
            if i == 0:
                if args.model_type == 'informer':
                    dec_inp = torch.zeros([batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).float().to(device)
                    dec_inp = torch.cat([batch_y[:,:args.label_len,:], dec_inp], dim=1).float()
                with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
                    if args.use_amp:
                        with torch.cuda.amp.autocast():
                            if args.model_type == 'informer':
                                outputs = model(batch_x, dec_inp) # (B,L,C)
                            else:
                                outputs = model(batch_x) # (B,L,C)
                    else:
                        if args.model_type == 'informer':
                            outputs = model(batch_x, dec_inp) # (B,L,C)
                        else:
                            outputs = model(batch_x) # (B,L,C)
                # print(prof.table())
                print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
                prof.export_chrome_trace(folder_path + '/profile.json')
                break
            
            # param
            if i == 0:
                if args.model_type == 'informer':
                    dec_inp = torch.zeros([batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).float().to(device)
                    dec_inp = torch.cat([batch_y[:,:args.label_len,:], dec_inp], dim=1).float()
                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        if args.model_type == 'informer':
                            flops, params = profile(model, inputs=(batch_x,dec_inp)) 
                        else:
                            flops, params = profile(model, inputs=(batch_x,))
                else:
                    if args.model_type == 'informer':
                        flops, params = profile(model, inputs=(batch_x,dec_inp))
                    else:
                        flops, params = profile(model, inputs=(batch_x,))

                flops, params = clever_format([flops, params], "%.3f")
                print("step{} | flops: {}, params: {}".format(i, flops, params))
                with open(folder_path + '/model_test.txt','a') as f:
                    f.writelines('step{} | flops: {}, params: {} \n'.format(i, flops, params))


            # if args.model_type == 'informer':
            #     dec_inp = torch.zeros([batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).float().to(device)
            #     dec_inp = torch.cat([batch_y[:,:args.label_len,:], dec_inp], dim=1).float()
            #     outputs = model(batch_x, dec_inp) # (B,L,C)
            # else:
            #     outputs = model(batch_x) # (B,L,C)

            # outputs = outputs.detach().cpu().numpy()
            # batch_y = batch_y.detach().cpu().numpy()

            # pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
            # true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

            # preds.append(pred)
            # trues.append(true)

            # fig = plot_heatmap_feature(outputs, batch_y)
            # plt.savefig(folder_path + '/heatmap_{}.jpg'.format(i))

            
    # preds = np.array(preds)
    # trues = np.array(trues)
    # print('test shape:', preds.shape, trues.shape)
    # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    # print('test shape:', preds.shape, trues.shape)

    # # result save
    # mae, mse, rmse, mape, mspe = metric(preds, trues)
    # print('mse:{}, mae:{}'.format(mse, mae))

    # # log metric and args setting
    # argsDict = args.__dict__
    # with open(folder_path + '/pred_result.txt','w') as f:
    #     f.write("metrics  \n")
    #     f.write('mse:{}, mae:{}'.format(mse, mae))
    #     f.write('\n')
    #     f.write('\n')
    #     f.writelines('Args  \n')
    #     for eachArg,value in argsDict.items():
    #         f.writelines(eachArg + ' : ' + str(value) + '\n')

    return

def get_inference_time():
    print('-------------------loading model-------------------')
    model.load_state_dict(torch.load(os.path.join(args.check_point, expname, 'valid_best_checkpoint.pth')))
    model.to(device)

    folder_path = os.path.join('./test_results', expname)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    model.eval()

    dummy_input = torch.randn(1, args.seq_len, args.enc_in,dtype=torch.float).to(device)
    dummy_out = torch.randn(1, args.label_len, args.enc_in,dtype=torch.float).to(device)
    if args.model_type == 'informer':
        dec_inp = torch.zeros([1, args.pred_len, args.enc_in], dtype=torch.float).to(device)
        dec_inp = torch.cat([dummy_out, dec_inp], dim=1).float()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    for _ in range(10):
        if args.model_type == 'informer':
            _ = model(dummy_input, dec_inp) # (B,L,C)
        else:
            _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.model_type == 'informer':
                        _ = model(dummy_input, dec_inp) # (B,L,C)
                    else:
                        _ = model(dummy_input) # (B,L,C)
            else:
                if args.model_type == 'informer':
                    _ = model(dummy_input, dec_inp) # (B,L,C)
                else:
                    _ = model(dummy_input) # (B,L,C)

            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn
    print(' * Mean {mean_syn:.3f}ms Std {std_syn:.3f}ms FPS {mean_fps:.2f}'.format(mean_syn=mean_syn, std_syn=std_syn, mean_fps=mean_fps))
    print(mean_syn)
    with open(folder_path + '/model_test.txt','a') as f:
        f.writelines(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn, std_syn=std_syn, mean_fps=mean_fps))




    

#main
train()
# test()
# pred()
# get_inference_time()





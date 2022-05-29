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
parser.add_argument('--resume',type=str, required=True, help='resume path')
parser.add_argument('--delta',type=float,default=0.05, help='threshold of occupancy')


args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

print('Args in experiment:')
print(args)

device = "cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu"

if args.model_type == 'TCN':
    num_channels = [args.hiden] * args.layer
    model = TCN(args.enc_in,num_channels,args.kernel,args.dropout,args.seq_len,args.pred_len)
    expname = '{}/{}/{}_att_in{}out{}_hiden{}_layer{}_lr{}_bs{}_epoch{}' \
        .format(args.data, args.model_type, \
            args.exp_name, args.seq_len, args.pred_len, args.hiden, args.layer, \
                args.learning_rate, args.batch_size, args.train_epochs)
elif args.model_type == 'MLP':
    model = MLP(args.seq_len,args.pred_len, middle=args.hiden)
    expname = '{}/{}/{}_att_in{}out{}_middle{}_lr{}_bs{}_epoch{}' \
        .format(args.data, args.model_type, \
            args.exp_name, args.seq_len, args.pred_len, args.hiden, \
                args.learning_rate, args.batch_size, args.train_epochs)
elif args.model_type == 'informer':
    model = Informer(args.enc_in, args.dec_in,args.c_out, args.seq_len, args.label_len,args.pred_len, \
        args.factor, args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.dropout, args.attn, \
        args.embed, args.freq, args.activation, args.output_attention, args.distil, args.mix, device)
    expname = '{}/{}/{}_att_in{}out{}_en{}de{}_dmodel{}dff{}_nhead{}_lr{}_bs{}_epoch{}' \
        .format(args.data, args.model_type, \
            args.exp_name, args.seq_len, args.pred_len, args.e_layers, args.d_layers, \
                args.d_model, args.d_layers, args.n_heads, \
                args.learning_rate, args.batch_size, args.train_epochs)
elif args.model_type == 'MLPMixer':
    model = MLPMixer(args.seq_len, args.pred_len, num_channels=args.enc_in, num_blocks=args.layer)
    expname = '{}/{}/{}_att_in{}out{}_lr{}_bs{}_blocks{}_epoch{}' \
        .format(args.data, args.model_type, \
            args.exp_name, args.seq_len, args.pred_len, \
                args.learning_rate, args.batch_size, args.layer, args.train_epochs)
elif args.model_type == 'rnn':
    model=RNNet(num_layers=args.rnn_layers,seq_len=args.seq_len,pred_len=args.pred_len)
    expname = '{}/{}/{}_att_in{}out{}_layer{}_lr{}_bs{}_epoch{}' \
        .format(args.data, args.model_type, \
            args.exp_name, args.seq_len, args.pred_len, args.rnn_layers, \
                args.learning_rate, args.batch_size, args.train_epochs)
elif args.model_type == 'lstm':
    model=LSTMnet(num_layers=args.rnn_layers,seq_len=args.seq_len,pred_len=args.pred_len)
    expname = '{}/{}/{}_att_in{}out{}_layer{}_lr{}_bs{}_epoch{}' \
        .format(args.data, args.model_type, \
            args.exp_name, args.seq_len, args.pred_len, args.rnn_layers, \
                args.learning_rate, args.batch_size, args.train_epochs)


print(model)
if args.resume is not None:
    print('loading model')
    model.load_state_dict(torch.load(args.resume))
    for p in model.parameters():
        p.requires_grad = False  # freeze model
model.to(device)
print(expname)



class Attack(nn.Module):
    def __init__(self, seq_len, pred_len, bs, enc_in, delta, device):
        super(Attack, self).__init__()

        self.perturbation = nn.Parameter(torch.zeros((bs,seq_len, enc_in), dtype=torch.float, device=device, requires_grad=True))
        self.adversarial_target = torch.zeros((bs, pred_len, enc_in), dtype=torch.float, device=device, requires_grad=False)
        self.func_clip = nn.Tanh()
        self.delta = delta

    def forward(self, x):
        return self.delta*self.func_clip(self.perturbation) + x


def vali(vali_data, vali_loader,att, criterion, epoch, writer, flag='vali'):
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

            batch_x = att(batch_x)
            
            if args.model_type == 'informer':
                dec_inp = torch.zeros([batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).float().to(device)
                dec_inp = torch.cat([batch_y[:,:args.label_len,:], dec_inp], dim=1).float()
                outputs = model(batch_x, dec_inp) # (B,L,C)
            else:
                outputs = model(batch_x) # (B,L,C)

            loss = (criterion(outputs, att.adversarial_target) - criterion(outputs, batch_y)).item()
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

    att = Attack(args.seq_len, args.pred_len, args.batch_size, args.enc_in, args.delta, device)
    
    optimizer = optim.Adam(att.parameters(),lr=args.learning_rate)
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

            batch_x = att(batch_x)

            if args.model_type == 'informer':
                dec_inp = torch.zeros([batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).float().to(device)
                dec_inp = torch.cat([batch_y[:,:args.label_len,:], dec_inp], dim=1).float()
                outputs = model(batch_x, dec_inp) # (B,L,C)
            else:
                outputs = model(batch_x) # (B,L,C)
            loss = criterion(outputs, att.adversarial_target) - criterion(outputs, batch_y)
            

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
        vali_loss, vali_ocpy_acc, vali_false_alarm, vali_missing_alarm = vali(vali_data, vali_loader, att, criterion, epoch, writer, 'vali')
        test_loss, test_ocpy_acc, test_false_alarm, test_missing_alarm = vali(test_data, test_loader, att, criterion, epoch, writer, 'test')

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
        
        
        adjust_learning_rate(optimizer, epoch + 1, args)

    return




#main
train()





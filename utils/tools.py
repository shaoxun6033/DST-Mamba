import numpy as np
from pyparsing import col
import torch
import matplotlib.pyplot as plt
import time
import pandas as pd

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    
def visual(true, preds=None, name='./pic/test.pdf'):
    """
    顶刊风格的时间序列结果可视化
    - Truth: 全程不带 marker，一条纯净实线
    - Pre: 仅在预测阶段带有 marker
    - 95%: 预测阶段的置信区间
    """
    true = np.asarray(true)
    
    # 1. 全局字体设置 (Times New Roman)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    
    # 2. 创建画布
    plt.figure(figsize=(6, 4), dpi=300)
    x = np.arange(len(true))
    
    # 3. 顶刊常用配色
    color_true = '#1A4D94'  # 深海蓝 (Truth)
    color_pred = '#CB3C33'  # 砖石红 (Pre)

    # 4. 绘制完整的真实值 (一条连续实线，无 marker，图例为 Truth)
    plt.plot(x, true, label='Truth', linewidth=2.5, color=color_true)

    if preds is not None:
        preds = np.asarray(preds)
        
        # --- 自动寻找预测数据的起始点 ---
        if len(preds) < len(true):
            split_idx = len(true) - len(preds)
            preds = np.concatenate([true[:split_idx], preds])
        else:
            diffs = np.abs(true - preds)
            diverge_indices = np.where(diffs > 1e-5)[0]
            split_idx = diverge_indices[0] if len(diverge_indices) > 0 else 0

        # 获取未来的坐标和预测值
        x_future = x[split_idx:]
        pred_future = preds[split_idx:]

        # ==========================================
        # 5. 绘制【预测部分】
        # ==========================================
        # 5.1 Pre 的未来走势 (红色，带空心方块 marker，图例为 Pre)
        plt.plot(x_future, pred_future, label='Predict', 
                 linewidth=2.5, color=color_pred,
                 marker='s', markersize=2.5, markerfacecolor='white', markeredgewidth=1.2)

        # 5.2 绘制置信区间 (图例为 95%)
        if len(pred_future) > 2:
            window_size = max(3, int(len(pred_future) / 10)) 
            preds_ser = pd.Series(pred_future)
            # 使用滑动窗口计算预测序列的标准差
            rolling_std = preds_ser.rolling(window=window_size, min_periods=1, center=True).std().values
            rolling_std = np.nan_to_num(rolling_std, nan=0.0) 
            
            margin = 1.96 * rolling_std
            plt.fill_between(x_future, pred_future - margin, pred_future + margin, 
                             color=color_pred, alpha=0.15, linewidth=0, 
                             label='95%')
                             
    # ==========================================
    # 6. 图表样式美化
    # ==========================================
    plt.grid(True, which='major', alpha=0.25, linestyle='--')
    
    ax = plt.gca()
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.2)
    
    plt.tick_params(axis='both', which='major', labelsize=12, width=1.2, length=5, direction='in')
    plt.xlabel('Time', fontsize=14, fontweight='bold')
    plt.ylabel('Values', fontsize=14, fontweight='bold')
    
    # 图例设置
    plt.legend(
    frameon=True,
    fontsize=11,
    loc='upper right',
    facecolor='white',   # 白色背景
    edgecolor='black'    # 黑色边框
)
    
    # 保存与展示
    plt.tight_layout()
    # os.makedirs(os.path.dirname(name), exist_ok=True) # 记得取消注释
    plt.savefig(name, format='pdf', bbox_inches='tight')
    plt.show()

# def visual(true, preds=None, name='./pic/test.pdf'):
#     """
#     Results visualization
#     """
#     plt.figure()
#     plt.plot(true, label='GroundTruth', linewidth=2,color='#1f77b4')
#     if preds is not None:
#         plt.plot(preds, label='Prediction', linewidth=2,color='#ff9b0e')
#     plt.grid(alpha=0.3, linestyle='--')
#     plt.legend()
#     plt.show()
#     plt.savefig(name, bbox_inches='tight')

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
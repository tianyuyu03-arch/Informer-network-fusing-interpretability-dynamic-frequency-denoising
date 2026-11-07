import argparse
from pathlib import Path
import numpy as np
from data_trans import data_trans
from myDataset_inf import MyDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import random
from model.model import My_Informer
import scipy.io as sio
import pandas as pd
# from utils.vmd import vmd

seed = 1  # 设置一个整数作为随机种子
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def evaluate(model, dataloader, criterion, outsize):
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    predictions = []
    truths = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device, dtype=torch.float32)
        batch_y = batch_y.to(device, dtype=torch.float32)
        pred_zeros = torch.zeros(
            batch_y.shape[0], model.pred_len, batch_y.shape[-1], device=device
        )
        dec_in = torch.cat((batch_y[:, :model.label_len, :], pred_zeros), dim=1)
        output = model(batch_x, dec_in).squeeze(1)
        target = batch_y[:, -model.pred_len:, -model.outsize:].squeeze(1)
        loss = criterion(output, target)
        batch_count = batch_x.shape[0]
        loss_sum += loss.item() * batch_count
        sample_count += batch_count
        predictions.append(output.cpu().numpy())
        truths.append(target.cpu().numpy())

    if sample_count == 0:
        raise ValueError('Evaluation split is empty.')
    return np.concatenate(truths), np.concatenate(predictions), loss_sum / sample_count


def train(model, dataloader, criterion, optimizer):
    model.train()
    loss_sum = 0.0
    sample_count = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device, dtype=torch.float32)
        batch_y = batch_y.to(device, dtype=torch.float32)
        pred_zeros = torch.zeros(
            batch_y.shape[0], model.pred_len, batch_y.shape[-1], device=device
        )
        dec_in = torch.cat((batch_y[:, :model.label_len, :], pred_zeros), dim=1)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch_x, dec_in).squeeze(1)
        target = batch_y[:, -model.pred_len:, -model.outsize:].squeeze(1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        batch_count = batch_x.shape[0]
        loss_sum += loss.item() * batch_count
        sample_count += batch_count

    if sample_count == 0:
        raise ValueError('Training split is empty.')
    return loss_sum / sample_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate the Informer model.')
    parser.add_argument('--data', type=Path, required=True, help='Path to a numeric CSV dataset kept outside version control.')
    parser.add_argument('--checkpoint', type=Path, default=Path('model_train/Train_Model.pth'), help='Path for the generated model weights.')
    parser.add_argument('--results-dir', type=Path, default=Path('run_result_MS'), help='Directory for generated evaluation results.')
    args = parser.parse_args()

    train_size = 3510
    validation_size = 390
    test_size = 100
    seq_len = 256
    label_len = 32
    pred_len = 1
    outsize = 1

    if not args.data.is_file():
        parser.error(f'dataset file does not exist: {args.data}')
    df = pd.read_csv(args.data)
    data = df.to_numpy(dtype=np.float64)
    required_rows = train_size + validation_size + test_size + seq_len + pred_len - 1
    if data.shape[1] < 2:
        parser.error('CSV must contain at least one feature column and one target column.')
    inpsize = data.shape[1] - outsize
    if len(data) < required_rows:
        parser.error(f'CSV must contain at least {required_rows} rows for the configured splits.')
    if not np.isfinite(data).all():
        parser.error('CSV contains missing or non-finite values.')

    target_scaler = MinMaxScaler()
    fit_end = train_size + seq_len + pred_len - 1
    data, target_scaler, input_scaler = data_trans(
        data, target_scaler, inpsize, outsize, fit_end
    )

    train_dataset = MyDataset(
        data, 'train', seq_len, label_len, pred_len,
        train_size, validation_size, test_size
    )
    validation_dataset = MyDataset(
        data, 'validation', seq_len, label_len, pred_len,
        train_size, validation_size, test_size
    )
    test_dataset = MyDataset(
        data, 'test', seq_len, label_len, pred_len,
        train_size, validation_size, test_size
    )
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=False)
    validation_dataloader = DataLoader(validation_dataset, batch_size=128, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    middlesize = 64
    d_ff = 64
    n_heads = 4
    encinsize = inpsize + outsize
    model_name = 'My_Informer'
    model = My_Informer(
        encinsize=encinsize, middlesize=middlesize, d_ff=d_ff,
        outsize=outsize, n_heads=n_heads, seq_len=seq_len,
        label_len=label_len, pred_len=pred_len
    ).to(device, dtype=torch.float32)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

    epochs = 50
    patience = 8
    best_validation_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    train_losses = []
    validation_losses = []
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer)
        _, _, validation_loss = evaluate(
            model, validation_dataloader, criterion, outsize
        )
        scheduler.step()
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        print(
            f'epoch {epoch + 1}/{epochs}: '
            f'train_mse={train_loss:.6f}, validation_mse={validation_loss:.6f}'
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.checkpoint)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f'Early stopping after epoch {epoch + 1}.')
                break

    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    truths, predictions, test_loss = evaluate(
        model, test_dataloader, criterion, outsize
    )
    print(
        f'Best epoch: {best_epoch}; validation MSE: {best_validation_loss:.6f}; '
        f'held-out test MSE: {test_loss:.6f}'
    )

    scaled_test_mse = test_loss
    truths = target_scaler.inverse_transform(truths)
    predictions = target_scaler.inverse_transform(predictions)
    original_test_mse = float(np.mean(np.square(truths - predictions)))
    original_test_mae = float(np.mean(np.abs(truths - predictions)))
    print(
        f'Test metrics in target units: MSE={original_test_mse:.6f}, '
        f'MAE={original_test_mae:.6f}; normalized MSE={scaled_test_mse:.6f}'
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.results_dir / 'My_Informer_VMD' / 'weight0'
    result_path.mkdir(parents=True, exist_ok=True)
    mat_name = args.data.stem
    sio.savemat(
        result_path / f'{model_name}_{mat_name}_test_result_inversed.mat',
        {'truth_y1': truths[:, 0], f'predict_y1_{model_name}': predictions[:, 0]},
    )
    sio.savemat(
        result_path / f'{model_name}_{mat_name}_loss.mat',
        {'train_loss': train_losses, 'validation_loss': validation_losses},
    )
    sio.savemat(
        args.results_dir / 'test_metrics.mat',
        {
            'test_mse_normalized': scaled_test_mse,
            'test_mse_original': original_test_mse,
            'test_mae_original': original_test_mae,
        },
    )
    np.savez_compressed(
        args.results_dir / f'{mat_name}_scalers.npz',
        input_mean=input_scaler.mean_, input_scale=input_scaler.scale_,
        target_min=target_scaler.min_, target_scale=target_scaler.scale_,
    )

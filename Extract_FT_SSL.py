import os
import argparse
import torch
import torchaudio
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from transformers import Wav2Vec2Model, Wav2Vec2Processor

class MosPredictor(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim):
        super().__init__()
        self.ssl_model = ssl_model
        self.output_estoi = nn.Linear(ssl_out_dim, 1)
        self.output_per = nn.Linear(ssl_out_dim, 1)
        self.output_wer = nn.Linear(ssl_out_dim, 1)

    def forward(self, wav, attention_mask=None):
        res = self.ssl_model(wav, attention_mask=attention_mask, return_dict=True)
        x = res.last_hidden_state  # [B, T, D]
        x = torch.mean(x, dim=1)   # [B, D]
        return (
            self.output_estoi(x).squeeze(1),
            self.output_per(x).squeeze(1),
            self.output_wer(x).squeeze(1),
            res,
        )

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, wavdir, label_file, processor):
        self.entries = []
        self.wavdir = wavdir
        self.processor = processor
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 4:
                    continue
                wavname, estoi, per, wer = parts
                self.entries.append((wavname, float(estoi), float(per), float(wer)))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        wavname, estoi, per, wer = self.entries[idx]
        wavpath = os.path.join(self.wavdir, wavname)
        wav, sr = torchaudio.load(wavpath)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        return wav.squeeze(0), estoi, per, wer, wavname

    def collate_fn(self, batch):
        wavs, estoi, per, wer, names = zip(*batch)
        wave_list = [w.numpy() for w in wavs]
        inputs = self.processor(wave_list, sampling_rate=16000, return_tensors="pt", padding=True)
        return (
            inputs['input_values'],
            inputs['attention_mask'],
            torch.tensor(estoi),
            torch.tensor(per),
            torch.tensor(wer),
            names
        )

def compute_metrics(pred, true):
    return {
        'R2': r2_score(true, pred),
        'PLCC': pearsonr(true, pred)[0],
        'SRCC': spearmanr(true, pred)[0]
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wavdir', type=str, required=True)
    parser.add_argument('--labelfile', type=str, required=True)
    parser.add_argument('--pretrained_model_path', type=str, required=True)
    parser.add_argument('--finetuned_ckpt', type=str, required=True)
    parser.add_argument('--ssl_save_dir', type=str, required=True)
    parser.add_argument('--out_csv', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.ssl_save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ssl_model = Wav2Vec2Model.from_pretrained(args.pretrained_model_path)
    processor = Wav2Vec2Processor.from_pretrained(args.pretrained_model_path)
    ssl_model.to(device)

    model = MosPredictor(ssl_model, ssl_model.config.hidden_size).to(device)
    model.load_state_dict(torch.load(args.finetuned_ckpt, map_location=device))
    model.eval()

    dataset = MyDataset(args.wavdir, args.labelfile, processor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)

    pred_estoi, pred_per, pred_wer = [], [], []
    true_estoi, true_per, true_wer = [], [], []
    filenames = []
    records = []
    with torch.no_grad():
        for input_values, attention_mask, y_estoi, y_per, y_wer, names in tqdm(loader):
            input_values = input_values.to(device)
            attention_mask = attention_mask.to(device)
            y_estoi = y_estoi.numpy()
            y_per = y_per.numpy()
            y_wer = y_wer.numpy()

            out_estoi, out_per, out_wer, res = model(input_values, attention_mask)
        
            out_estoi = out_estoi.cpu().numpy()
            out_per = out_per.cpu().numpy()
            out_wer = out_wer.cpu().numpy()
            true_estoi.extend(y_estoi)
            true_per.extend(y_per)
            true_wer.extend(y_wer)
            pred_estoi.extend(out_estoi)
            pred_per.extend(out_per)
            pred_wer.extend(out_wer)
            filenames.extend(names)
            embeddings = res.last_hidden_state.mean(dim=1).cpu().numpy()
            ##embeddings = res.last_hidden_state.squeeze(0).cpu().numpy()
            npy_name = os.path.splitext(os.path.basename(names[0]))[0] + '.npy'
            np.save(os.path.join(args.ssl_save_dir, npy_name), embeddings)
            ##for name, gt_estoi, gt_per, gt_wer, pr_estoi, pr_per, pr_wer, emb in zip(names, y_estoi, y_per, y_wer, p_estoi, p_per, p_wer, embeddings):
                
                ##npy_path = os.path.join(args.ssl_save_dir, name + '.npy')
                ##np.save(npy_path, emb)
                ##records.append([name, gt_estoi, pr_estoi, gt_per, pr_per, gt_wer, pr_wer])

    ##df = pd.DataFrame(records, columns=["filename", "GT_ESTOI", "PR_ESTOI", "GT_PER", "PR_PER", "GT_WER", "PR_WER"])
    print("Lengths:")
    print(f"filename: {len(filenames)}")
    print(f"true_ESTOI: {len(true_estoi)}  pred_ESTOI: {len(pred_estoi)}")
    print(f"true_PER: {len(true_per)}  pred_PER: {len(pred_per)}")
    print(f"true_WER: {len(true_wer)}  pred_WER: {len(pred_wer)}")
    # 保存 CSV
    df = pd.DataFrame({
        "filename": filenames,
        "GT_ESTOI": true_estoi,
        "PR_ESTOI": pred_estoi,
        "GT_PER": true_per,
        "PR_PER": pred_per,
        "GT_WER": true_wer,
        "PR_WER": pred_wer,
    })
    df.to_csv(args.out_csv, index=False)
    print(f"✅ CSV saved to {args.out_csv}")

     # 输出指标
    def metrics(y_true, y_pred, name):
        r2 = r2_score(y_true, y_pred)
        plcc = pearsonr(y_true, y_pred)[0]
        srcc = spearmanr(y_true, y_pred)[0]
        print(f"\n{name} → R2: {r2:.4f}, PLCC: {plcc:.4f}, SRCC: {srcc:.4f}")

    metrics(np.array(true_estoi), np.array(pred_estoi), "ESTOI")
    metrics(np.array(true_per), np.array(pred_per), "PER")
    metrics(np.array(true_wer), np.array(pred_wer), "WER")

if __name__ == '__main__':
    main()

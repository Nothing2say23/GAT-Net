import os
import argparse
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
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

class MyDataset(Dataset):
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
                # 注意：顺序是 ESTOI, PER, WER
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
        wave_list = [w.numpy() for w in wavs]  # for processor
        inputs = self.processor(
            wave_list, sampling_rate=16000, return_tensors="pt", padding=True
        )
        return (
            inputs['input_values'],
            inputs['attention_mask'],
            torch.tensor(estoi),
            torch.tensor(per),
            torch.tensor(wer),
            names
        )

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_values, attn_mask, y_wer, y_per, y_stoi, _ = [b.to(device) if torch.is_tensor(b) else b for b in batch]
            out_wer, out_per, out_stoi, _ = model(input_values, attn_mask)
            loss = criterion(out_wer, y_wer) + criterion(out_per, y_per) + criterion(out_stoi, y_stoi)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wavdir', required=True)
    parser.add_argument('--train_label', required=True)
    parser.add_argument('--val_label', required=True)
    parser.add_argument('--pretrained_model_path', required=True)
    parser.add_argument('--finetune_from_checkpoint', required=True)  # 必须指定已有参数
    parser.add_argument('--outdir', type=str, default='./output')
    parser.add_argument('--total_epochs', type=int, default=30)  # 再训练 30 次
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ 加载预训练 + 微调模型
    ssl_model = Wav2Vec2Model.from_pretrained(args.pretrained_model_path).to(device)
    processor = Wav2Vec2Processor.from_pretrained(args.pretrained_model_path)
    net = MosPredictor(ssl_model, ssl_model.config.hidden_size).to(device)

    # ✅ 加载之前保存的最优模型参数
    print(f"Loading fine-tuned checkpoint from: {args.finetune_from_checkpoint}")
    net.load_state_dict(torch.load(args.finetune_from_checkpoint, map_location=device))

    # ✅ 数据集准备
    train_set = MyDataset(args.wavdir, args.train_label, processor)
    val_set = MyDataset(args.wavdir, args.val_label, processor)
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2, collate_fn=train_set.collate_fn)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=2, collate_fn=val_set.collate_fn)

    # ✅ 优化器（重新初始化即可）
    criterion = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    # ✅ 新阶段的训练控制
    best_val_loss = float('inf')
    patience = 10
    stop_count = 0

    for epoch in range(1, args.total_epochs + 1):
        net.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            input_values, attn_mask, y_estoi, y_per, y_wer, _ = [b.to(device) if torch.is_tensor(b) else b for b in batch]
            optimizer.zero_grad()
            out_estoi, out_per, out_wer, _ = net(input_values, attn_mask)
            loss = criterion(out_estoi, y_estoi) + criterion(out_per, y_per) + criterion(out_wer, y_wer)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss = evaluate(net, val_loader, criterion, device)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.outdir, f"best_model_cont_epoch{epoch}.pt")
            torch.save(net.state_dict(), save_path)
            print(f"✅ Saved best model to {save_path}")
            stop_count = 0
        else:
            stop_count += 1
            if stop_count >= patience:
                print(f"⏹ Early stopping at epoch {epoch}")
                break

if __name__ == "__main__":
    main()

import os
import cv2
import time
import csv
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import albumentations as A
from albumentations.pytorch import ToTensorV2
from effdet import create_model, DetBenchTrain, DetBenchPredict

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
set_seed()

class Config:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    ANNOTATIONS_DIR = os.path.join(BASE_PATH, 'annotations')
    IMAGES_ROOT = os.path.join(BASE_PATH, 'images')

    TRAIN_IMG_DIR = os.path.join(IMAGES_ROOT, 'Train')
    VAL_IMG_DIR = os.path.join(IMAGES_ROOT, 'Validation')
    TRAIN_JSON = os.path.join(ANNOTATIONS_DIR, 'instances_Train.json')
    VAL_JSON = os.path.join(ANNOTATIONS_DIR, 'instances_Validation.json')
    
    IMG_SIZE = 640
    BATCH_SIZE = 8
    ACCUMULATION_STEPS = 2
    PATIENCE = 50
    EPOCHS = 200
    LR = 5e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    OUTPUT_MODEL_BEST = os.path.join(BASE_PATH, "effdet_best.pth")
    OUTPUT_MODEL_LAST = os.path.join(BASE_PATH, "effdet_last.pth")
    CHECKPOINT_LAST = os.path.join(BASE_PATH, "effdet_last_checkpoint.pth")
    CSV_LOG_FILE = os.path.join(BASE_PATH, "training_results_effdet.csv")
    
    RESUME_TRAINING = True

    CSV_HEADERS = [
        "Epoch", "Train_Loss", "Val_Loss", "LR", "Time_Sec",
        "mAP_0.50:0.95_All", "mAP_0.50_All", "mAP_0.75_All",
        "mAP_Small", "mAP_Medium", "mAP_Large",
        "AR_1", "AR_10", "AR_100", "AR_Small", "AR_Medium", "AR_Large"
    ]

def format_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

class LegoDataset(Dataset):
    def __init__(self, img_dir, ann_path):
        self.coco = COCO(ann_path)
        self.img_dir = img_dir
        self.ids = list(self.coco.imgs.keys())
        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat2idx = {cat_id: i + 1 for i, cat_id in enumerate(self.cat_ids)}
        self.idx2cat = {i + 1: cat_id for i, cat_id in enumerate(self.cat_ids)}
        
        self.transform = A.Compose([
            A.LongestMaxSize(max_size=Config.IMG_SIZE),
            A.PadIfNeeded(min_height=Config.IMG_SIZE, min_width=Config.IMG_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        image = cv2.imread(os.path.join(self.img_dir, img_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w > 1 and h > 1:
                boxes.append([x, y, x + w, y + h])
                labels.append(self.cat2idx[ann['category_id']])

        if len(boxes) > 0:
            sample = self.transform(image=image, bboxes=boxes, labels=labels)
            image = sample['image']
            boxes = np.array(sample['bboxes'])
            if len(boxes) > 0:
                boxes = boxes[:, [1, 0, 3, 2]] 
            labels = np.array(sample['labels'], dtype=np.int64)
        else:
            sample = self.transform(image=image, bboxes=[], labels=[])
            image = sample['image']
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.array([], dtype=np.int64)

        return image, {
            'bboxes': torch.as_tensor(boxes, dtype=torch.float32), 
            'labels': torch.as_tensor(labels, dtype=torch.int64), 
            'img_id': torch.tensor([img_id])
        }

    def __len__(self):
        return len(self.ids)

def collate_fn(batch):
    return tuple(zip(*batch))

def evaluate(eval_bench, train_bench, data_loader, device):
    eval_bench.eval()
    train_bench.eval()
    coco_gt = data_loader.dataset.coco
    coco_pred_results = []
    total_val_loss = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = torch.stack(images).to(device, non_blocking=True)
            
            max_boxes = max([t['bboxes'].shape[0] for t in targets])
            batch_boxes = torch.full((len(targets), max_boxes, 4), -1, dtype=torch.float32, device=device)
            batch_labels = torch.full((len(targets), max_boxes), -1, dtype=torch.float32, device=device)
            for i, t in enumerate(targets):
                n = t['bboxes'].shape[0]
                if n > 0:
                    batch_boxes[i, :n], batch_labels[i, :n] = t['bboxes'], t['labels']
            
            bench_targets = {
                'bbox': batch_boxes,
                'cls': batch_labels,
                'img_scale': torch.ones(len(targets), device=device),
                'img_size': torch.tensor([(Config.IMG_SIZE, Config.IMG_SIZE)] * len(targets), device=device, dtype=torch.float32)
            }
            
            loss_output = train_bench(images, bench_targets)
            total_val_loss += loss_output['loss'].item()

            outputs = eval_bench(images)
            for i, out in enumerate(outputs):
                if out.numel() == 0: continue
                img_id = targets[i]['img_id'].item()
                img_info = coco_gt.loadImgs(img_id)[0]
                
                scale = max(img_info['height'], img_info['width']) / Config.IMG_SIZE
                pad_y = (Config.IMG_SIZE - int(img_info['height'] / scale)) // 2
                pad_x = (Config.IMG_SIZE - int(img_info['width'] / scale)) // 2
                
                for b, s, l in zip(out[:, :4].cpu().numpy(), out[:, 4].cpu().numpy(), out[:, 5].cpu().numpy()):
                    x_min_orig = max(0, (b[0] - pad_x) * scale)
                    y_min_orig = max(0, (b[1] - pad_y) * scale)
                    w_orig, h_orig = (b[2] - b[0]) * scale, (b[3] - b[1]) * scale
                    
                    coco_pred_results.append({
                        'image_id': img_id, 
                        'category_id': data_loader.dataset.idx2cat[int(l)],
                        'bbox': [float(x_min_orig), float(y_min_orig), float(w_orig), float(h_orig)], 
                        'score': float(s)
                    })
                    
    avg_val_loss = total_val_loss / len(data_loader)
    if not coco_pred_results: 
        return [0.0] * 12, avg_val_loss
    
    coco_dt = coco_gt.loadRes(coco_pred_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    
    return coco_eval.stats, avg_val_loss


def main():
    use_amp = torch.cuda.is_available()
    
    train_ds = LegoDataset(Config.TRAIN_IMG_DIR, Config.TRAIN_JSON)
    val_ds = LegoDataset(Config.VAL_IMG_DIR, Config.VAL_JSON)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=4, 
        pin_memory=True
    )
    
    model = create_model(
        'tf_efficientdet_d1', 
        pretrained=True, 
        num_classes=len(train_ds.cat_ids), 
        image_size=(Config.IMG_SIZE, Config.IMG_SIZE)
    )
    train_bench = DetBenchTrain(model).to(Config.DEVICE)
    eval_bench = DetBenchPredict(model).to(Config.DEVICE)

    optimizer = torch.optim.AdamW(train_bench.parameters(), lr=Config.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    
    start_epoch = 0
    best_map = 0
    epochs_without_improvement = 0 

    if Config.RESUME_TRAINING and os.path.exists(Config.CHECKPOINT_LAST):
        ckpt = torch.load(Config.CHECKPOINT_LAST, map_location=Config.DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch']
        best_map = ckpt['best_map']
        epochs_without_improvement = ckpt.get('epochs_without_improvement', 0)
        print(f"--> Wznowiono od epoki {start_epoch+1} (Best mAP: {best_map:.4f}, Bez poprawy od: {epochs_without_improvement} epok)")

    if start_epoch == 0:
        with open(Config.CSV_LOG_FILE, mode='w', newline='') as f:
            csv.writer(f).writerow(Config.CSV_HEADERS)

    for epoch in range(start_epoch, Config.EPOCHS):
        epoch_start = time.time()
        train_bench.train()
        total_loss = 0
        optimizer.zero_grad()
        
        for step, (images, targets) in enumerate(train_loader):
            images = torch.stack(images).to(Config.DEVICE, non_blocking=True)

            max_boxes = max([t['bboxes'].shape[0] for t in targets])
            batch_boxes = torch.full((len(targets), max_boxes, 4), -1, dtype=torch.float32, device=Config.DEVICE)
            batch_labels = torch.full((len(targets), max_boxes), -1, dtype=torch.float32, device=Config.DEVICE)
            
            for i, t in enumerate(targets):
                n = t['bboxes'].shape[0]
                if n > 0:
                    batch_boxes[i, :n], batch_labels[i, :n] = t['bboxes'], t['labels']
            
            train_targets = {
                'bbox': batch_boxes,
                'cls': batch_labels,
                'img_scale': torch.ones(len(targets), device=Config.DEVICE),
                'img_size': torch.tensor([(Config.IMG_SIZE, Config.IMG_SIZE)] * len(targets), device=Config.DEVICE, dtype=torch.float32)
            }
            
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                loss = train_bench(images, train_targets)['loss']
                loss = loss / Config.ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (step + 1) % Config.ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(train_bench.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total_loss += loss.item() * Config.ACCUMULATION_STEPS

        avg_loss = total_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        epoch_dur = time.time() - epoch_start
        
        all_stats = ["N/A"] * 12
        val_loss_str = "N/A"
        is_eval_epoch = True
        
        if is_eval_epoch:
            print(f"\n--- Ewaluacja (Epoka {epoch+1}) ---")
            stats, v_loss = evaluate(eval_bench, train_bench, val_loader, Config.DEVICE)
            val_loss_str = f"{v_loss:.4f}"
            all_stats = [f"{s:.4f}" for s in stats]
            
            current_map = stats[0]
            
            if current_map > best_map:
                best_map = current_map
                epochs_without_improvement = 0  
                torch.save(model.state_dict(), Config.OUTPUT_MODEL_BEST)
                print(f"✔ Nowy rekord mAP: {best_map:.4f}! Zapisano model best.")
            else:
                epochs_without_improvement += 1
                print(f"✘ Brak poprawy (Best mAP: {best_map:.4f}). Licznik cierpliwości: {epochs_without_improvement}/{Config.PATIENCE}")

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_map': best_map,
            'epochs_without_improvement': epochs_without_improvement
        }, Config.CHECKPOINT_LAST)

        torch.save(model.state_dict(), Config.OUTPUT_MODEL_LAST)
        print(f"Epoch {epoch+1}/{Config.EPOCHS} | Loss: {avg_loss:.4f} | Val_Loss: {val_loss_str} | mAP: {all_stats[0]} | Time: {format_time(epoch_dur)}")

        with open(Config.CSV_LOG_FILE, mode='a', newline='') as f:
            csv.writer(f).writerow([epoch+1, f"{avg_loss:.4f}", val_loss_str, f"{current_lr:.6f}", f"{epoch_dur:.2f}"] + all_stats)

        if epochs_without_improvement >= Config.PATIENCE:
            print(f"\n[!] EARLY STOPPING: Brak poprawy przez {Config.PATIENCE} epok. Kończenie treningu.")
            break

if __name__ == "__main__":
    main()

# ==================================================================================
# 0. IMPORTACIONES Y CONFIGURACIÓN
# ==================================================================================
import os
import io
import gc
import random
import time
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from torchvision.transforms import InterpolationMode

from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
from sklearn.utils import resample

path = "./data/breakhis_npz"
files = [f for f in os.listdir(path) if f.endswith(".npz")]

rows = []
for f in files:
    parts = f.replace(".npz", "").split("_")
    ttype = parts[0]
    pid = parts[-1]
    subtype = "_".join(parts[1:-1])  # combina lo que hay entre medio
    arr = np.load(os.path.join(path, f))
    imgs = arr[list(arr.keys())[0]]
    rows.append((ttype, subtype, pid, imgs.shape[0]))

df = pd.DataFrame(rows, columns=["TumorType","Subtype","PatientID","NumImages"])

# 1. Definimos el splitter que RESPETA grupos (Pacientes)
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=3)

# 2. Generamos índices usando los grupos
# Observa que pasamos 'groups=df["PatientID"]'
train_val_idx, test_idx = next(gss.split(df, df["TumorType"], groups=df["PatientID"]))

train_val = df.iloc[train_val_idx]
test = df.iloc[test_idx]

# 3. Repetimos para separar Train / Val (dentro de train_val)
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=3)

train_idx, val_idx = next(gss2.split(train_val, train_val["TumorType"], groups=train_val["PatientID"]))

train = train_val.iloc[train_idx]
val = train_val.iloc[val_idx]


# Configuración dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(SEED)
print(f"✅ Entorno listo. Dispositivo: {device} | Seed: {SEED}")

# ==================================================================================
# 1. CLASES AUXILIARES (EarlyStopping, Augmentations, Dataset)
# ==================================================================================

class EarlyStopping:
    def __init__(self, patience=7, delta=0.001, mode='max'):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.mode == 'max':
            improvement = current_score - self.best_score
        else:
            improvement = self.best_score - current_score

        if improvement > self.delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

# --- Transformaciones ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.03):
        self.mean = mean
        self.std = std
    def __call__(self, img):
        img_np = np.array(img, dtype=np.float32) / 255.0
        noise = np.random.normal(self.mean, self.std, img_np.shape)
        noisy_img = np.clip(img_np + noise, 0, 1)
        return Image.fromarray((noisy_img * 255).astype(np.uint8))

class RandomRotate90(object):
    def __init__(self, angles=[90, 180, 270]):
        self.angles = angles
    def __call__(self, img):
        angle = random.choice(self.angles)
        return T.functional.rotate(img, angle)

def get_transforms(scenario):
    val_transform = T.Compose([
        T.Resize(224, interpolation=InterpolationMode.BILINEAR),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    # Bloques constructivos
    geom_ops = [
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomApply([RandomRotate90()], p=0.5),
        T.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(1.0, 1.0), interpolation=InterpolationMode.BILINEAR)
    ]
    color_op = T.RandomApply([
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05)
    ], p=0.5)
    elastic_op = T.RandomApply([
        T.ElasticTransform(alpha=22.4, sigma=22.4, interpolation=InterpolationMode.BILINEAR)
    ], p=0.5)
    noise_op = T.RandomApply([AddGaussianNoise(std=0.03)], p=0.5)

    if scenario == "None":
        train_transform = T.Compose([
            T.Resize(224, interpolation=InterpolationMode.BILINEAR),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    elif scenario == "Basic":
        train_transform = T.Compose(geom_ops + [color_op, T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    elif scenario == "Advanced":
        train_transform = T.Compose(geom_ops + [color_op, elastic_op, noise_op, T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    else:
        raise ValueError("Scenario Unknown")
    return train_transform, val_transform

# --- Dataset & Utils ---
def build_image_index(df_split, base_path):
    samples = []
    for _, row in df_split.iterrows():
        fname = f"{row.TumorType}_{row.Subtype}_{row.PatientID}.npz"
        full_path = os.path.join(base_path, fname)
        label = 0 if row.TumorType == "benign" else 1
        num_imgs = int(row.NumImages)
        for tile_idx in range(num_imgs):
            samples.append((full_path, tile_idx, label))
    return samples

class BreakHisTilesDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        fname, tile_idx, label = self.samples[idx]
        try:
            arr = np.load(fname)
            key = "images" if "images" in arr.files else arr.files[0]
            img = arr[key][tile_idx]
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)
            if self.transform: img = self.transform(img)
            return img, torch.tensor(label, dtype=torch.float32)
        except:
            return torch.zeros((3, 224, 224)), torch.tensor(label, dtype=torch.float32)

def create_model(hparams):
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in resnet.parameters(): param.requires_grad = False
    resnet.fc = nn.Sequential(
        nn.Linear(resnet.fc.in_features, hparams["head_hidden"]),
        nn.ReLU(inplace=True),
        nn.Dropout(hparams["dropout"]),
        nn.Linear(hparams["head_hidden"], 1)
    )
    return resnet.to(device)

def evaluate_model(loader, model):
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            outputs = model(imgs).squeeze(1)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(targets.numpy())
    try: return roc_auc_score(all_targets, all_probs)
    except: return 0.5

# ==================================================================================
# 2. CONFIGURACIÓN DE EJECUCIÓN (DEBUG vs REAL)
# ==================================================================================
# 🔴 INTERRUPTOR 
DEBUG_MODE = False  # <--- CAMBIAR A 'FALSE' PARA EL ENTRENAMIENTO DEFINITIVO

if DEBUG_MODE:
    print("\n⚠️ ATENCIÓN: MODO DEBUG ACTIVADO")
    print("   - Usando subset muy pequeño.")
    print("   - Pocas épocas/trials.")
    print("   - 🚫 NO SE GUARDARÁN ARCHIVOS EN DISCO.")
    
    # Subset rápido
    train_subset = train.sample(n=15, random_state=SEED) 
    val_subset   = val.sample(n=10, random_state=SEED)
    df_train_phase1, df_val_phase1 = train_subset, val_subset
    
    N_TRIALS = 2        
    N_EPOCHS_RS = 1     
    N_EPOCHS_CV = 2     
    PATIENCE_CV = 2
    K_FOLDS = 2
else:
    print("\n🚀 MODO REAL ACTIVADO")
    print("   - Usando dataset completo.")
    print("   - Configuración completa.")
    print("   - ✅ SE GUARDARÁN MODELOS Y RESULTADOS.")
    
    df_train_phase1, df_val_phase1 = train, val
    
    N_TRIALS = 15       # 15 trials para explorar bien
    N_EPOCHS_RS = 6     # Cribado rápido
    N_EPOCHS_CV = 30    # Validación sólida
    PATIENCE_CV = 7     # Early Stopping estándar
    K_FOLDS = 3         # K=3 por paciente

FIXED_BATCH_SIZE = 64
train_idx_p1 = build_image_index(df_train_phase1, path)
val_idx_p1   = build_image_index(df_val_phase1, path)

n_pos = sum(1 for _, _, y in train_idx_p1 if y == 1)
n_neg = sum(1 for _, _, y in train_idx_p1 if y == 0)
POS_WEIGHT = torch.tensor(n_neg / n_pos, dtype=torch.float32).to(device)


# ==================================================================================
# 3. FASE 1: RANDOM SEARCH (Selección Top 3)
# ==================================================================================
HP_SPACE = {
    "lr": [2e-4, 5e-4, 1e-3],    
    "weight_decay": [1e-4, 1e-3],
    "head_hidden": [128, 256, 512], 
    "dropout": [0.3, 0.5, 0.7]
}
SCENARIOS = ["None", "Basic", "Advanced"]
rs_results = []

print(f"\n{'='*40}\n INICIANDO FASE 1: RANDOM SEARCH \n{'='*40}")

for scenario in SCENARIOS:
    print(f"\n🔹 Escenario: {scenario}")
    train_tf, val_tf = get_transforms(scenario)
    ds_val = BreakHisTilesDataset(val_idx_p1, transform=val_tf)
    
    for trial_i in range(N_TRIALS):
        hp = {k: random.choice(v) for k, v in HP_SPACE.items()}
        
        ds_train = BreakHisTilesDataset(train_idx_p1, transform=train_tf)
        dl_train = DataLoader(ds_train, batch_size=FIXED_BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
        dl_val   = DataLoader(ds_val,   batch_size=FIXED_BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)
        
        model = create_model(hp)
        optimizer = optim.AdamW(model.fc.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        criterion = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS_RS)
        
        best_auc = 0.0
        for epoch in range(N_EPOCHS_RS):
            model.train()
            for imgs, targets in dl_train:
                imgs, targets = imgs.to(device), targets.to(device)
                optimizer.zero_grad()
                loss = criterion(model(imgs).squeeze(1), targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            auc_val = evaluate_model(dl_val, model)
            if auc_val > best_auc: best_auc = auc_val
        
        print(f"   Trial {trial_i+1}/{N_TRIALS} | AUC: {best_auc:.4f} | {hp}")
        rs_results.append({**hp, "scenario": scenario, "val_auc": best_auc})
        del model, optimizer; torch.cuda.empty_cache(); gc.collect()

df_rs = pd.DataFrame(rs_results)
df_top3 = df_rs.groupby("scenario").apply(lambda x: x.nlargest(3, 'val_auc')).reset_index(drop=True)
print("\n🏆 TOP 3 CANDIDATOS SELECCIONADOS:")
print(df_top3[["scenario", "val_auc", "lr", "head_hidden"]])


# ==================================================================================
# 4. FASE 2: CROSS-VALIDATION INTERNA (K=3)
# ==================================================================================
print(f"\n{'='*40}\n INICIANDO FASE 2: CV INTERNA SOBRE TOP 3 \n{'='*40}")

df_full = pd.concat([df_train_phase1, df_val_phase1]).reset_index(drop=True)
y_vals = df_full["TumorType"].apply(lambda x: 1 if x == 'malign' else 0).values
groups = df_full["PatientID"].values
sgkf = StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

cv_results = []

for idx, cand in df_top3.iterrows():
    scenario = cand["scenario"]
    hp = cand.to_dict()
    cand_id = f"{scenario}_Cand{idx}"
    print(f"\n🔶 Evaluando: {cand_id} (lr={hp['lr']}, hidden={hp['head_hidden']})")
    
    fold_aucs = []
    for fold, (t_idx, v_idx) in enumerate(sgkf.split(df_full, y_vals, groups=groups)):
        print(f"   > Fold {fold+1}/{K_FOLDS}...", end="")
        
        train_tf, val_tf = get_transforms(scenario)
        idx_t = build_image_index(df_full.iloc[t_idx], path)
        idx_v = build_image_index(df_full.iloc[v_idx], path)
        
        dl_t = DataLoader(BreakHisTilesDataset(idx_t, transform=train_tf), batch_size=FIXED_BATCH_SIZE, shuffle=True, num_workers=6)
        dl_v = DataLoader(BreakHisTilesDataset(idx_v, transform=val_tf), batch_size=FIXED_BATCH_SIZE, shuffle=False, num_workers=6)
        
        model = create_model(hp)
        optimizer = optim.AdamW(model.fc.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        criterion = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT)
        stopper = EarlyStopping(patience=PATIENCE_CV, mode='max')
        
        best_f_auc = 0.0
        for epoch in range(N_EPOCHS_CV):
            model.train()
            for imgs, targets in dl_t:
                imgs, targets = imgs.to(device), targets.to(device)
                optimizer.zero_grad()
                loss = criterion(model(imgs).squeeze(1), targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            auc_v = evaluate_model(dl_v, model)
            if auc_v > best_f_auc: best_f_auc = auc_v
            if stopper(auc_v): break
        
        print(f" Best: {best_f_auc:.4f}")
        fold_aucs.append(best_f_auc)
        del model, optimizer; torch.cuda.empty_cache(); gc.collect()

    cv_results.append({"Scenario": scenario, "Mean_AUC": np.mean(fold_aucs), "HP": hp})

df_cv = pd.DataFrame(cv_results)
df_winners = df_cv.loc[df_cv.groupby("Scenario")["Mean_AUC"].idxmax()].reset_index(drop=True)
print("\n🎉 GANADORES FINALES POR ESCENARIO:")
print(df_winners[["Scenario", "Mean_AUC"]])


# ==================================================================================
# 5. FASE 3: EVALUACIÓN EN TEST (BOOTSTRAP + YOUDEN)
# ==================================================================================
print(f"\n{'='*40}\n INICIANDO FASE 3: TEST FINAL \n{'='*40}")

# Unimos todo train+val original
df_train_final = pd.concat([df_train_phase1, df_val_phase1]).reset_index(drop=True)
# Test intocable
df_test_final = test 

test_final_idx = build_image_index(df_test_final, path)
POS_WEIGHT_FINAL = torch.tensor(n_neg / n_pos, dtype=torch.float32).to(device)

final_predictions = {}
test_metrics = []

def compute_bootstrap_ci(y_true, y_pred, threshold, n_boot=1000):
    stats = {'sens':[], 'spec':[], 'ppv':[], 'npv':[]}
    y_pred_bin = (y_pred >= threshold).astype(int)
    for _ in range(n_boot):
        indices = resample(np.arange(len(y_true)), replace=True)
        yt, yp = y_true[indices], y_pred_bin[indices]
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0,1]).ravel()
        stats['sens'].append(tp/(tp+fn) if (tp+fn)>0 else 0)
        stats['spec'].append(tn/(tn+fp) if (tn+fp)>0 else 0)
        stats['ppv'].append(tp/(tp+fp) if (tp+fp)>0 else 0)
        stats['npv'].append(tn/(tn+fn) if (tn+fn)>0 else 0)
    res = {}
    for k, v in stats.items(): res[k] = (np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5))
    return res

for _, row in df_winners.iterrows():
    scenario = row["Scenario"]
    hp = row["HP"]
    print(f"\n🚀 Entrenando Final: {scenario}...")
    
    # Split interno 90/10 para Early Stopping
    gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
    t_idx, v_idx = next(gss.split(df_train_final, groups=df_train_final["PatientID"]))
    
    idx_t = build_image_index(df_train_final.iloc[t_idx], path)
    idx_v = build_image_index(df_train_final.iloc[v_idx], path)
    train_tf, val_tf = get_transforms(scenario)
    
    dl_t = DataLoader(BreakHisTilesDataset(idx_t, transform=train_tf), batch_size=FIXED_BATCH_SIZE, shuffle=True, num_workers=6)
    dl_v = DataLoader(BreakHisTilesDataset(idx_v, transform=val_tf), batch_size=FIXED_BATCH_SIZE, shuffle=False, num_workers=6)
    dl_test = DataLoader(BreakHisTilesDataset(test_final_idx, transform=val_tf), batch_size=FIXED_BATCH_SIZE, shuffle=False, num_workers=6)
    
    model = create_model(hp)
    optimizer = optim.AdamW(model.fc.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT_FINAL)
    stopper = EarlyStopping(patience=PATIENCE_CV, mode='max')
    
    EPOCHS_F3 = 2 if DEBUG_MODE else 30
    best_auc_int = 0.0
    
    # Training Loop Final
    for epoch in range(EPOCHS_F3):
        model.train()
        for imgs, targets in dl_t:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs).squeeze(1), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        auc_v = evaluate_model(dl_v, model)
        
        # GUARDADO DE MODELO (Solo si NO es debug)
        if auc_v > best_auc_int:
            best_auc_int = auc_v
            if not DEBUG_MODE:
                torch.save(model.state_dict(), f"final_model_{scenario}.pth")
        
        if stopper(auc_v): break

    # Inferencia en Test
    if not DEBUG_MODE:
        model.load_state_dict(torch.load(f"final_model_{scenario}.pth"))
    
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for imgs, targets in dl_test:
            imgs = imgs.to(device)
            outs = model(imgs).squeeze(1)
            probs = torch.sigmoid(outs).cpu().numpy()
            y_prob.extend(probs); y_true.extend(targets.numpy())
            
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    final_predictions[scenario] = {"y_true": y_true, "y_prob": y_prob}
    
    # Métricas
    fpr, tpr, ths = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    best_thresh = ths[np.argmax(tpr - fpr)] # Youden
    
    print(f"   -> AUC Test: {roc_auc:.4f} | Youden Thresh: {best_thresh:.4f}")
    
    if not DEBUG_MODE:
        print("   -> Bootstraping...")
        cis = compute_bootstrap_ci(y_true, y_prob, best_thresh)
        test_metrics.append({
            "Scenario": scenario, "AUC": roc_auc, "Thresh": best_thresh,
            "Sens": cis['sens'], "Spec": cis['spec'], "PPV": cis['ppv'], "NPV": cis['npv']
        })

# ==================================================================================
# 6. GUARDADO DE RESULTADOS
# ==================================================================================
if not DEBUG_MODE:
    print("\n💾 GUARDANDO RESULTADOS EN DISCO...")
    
    # Tabla Fase 1 (RS)
    df_rs.to_csv("tfm_p1_random_search.csv", index=False)
    
    # Tabla Fase 2 (CV)
    df_cv.to_csv("tfm_p2_internal_cv.csv", index=False)
    
    # Tabla Fase 3 (Test Metrics)
    pd.DataFrame(test_metrics).to_csv("tfm_test_metrics.csv", index=False)
    
    # Predicciones Crudas (Numpy)
    np.save("tfm_test_predictions.npy", final_predictions)
    
    print("✅ Todo guardado: .csv, .npy y .pth listos para descargar.")
else:
    print("\n🛑 MODO DEBUG: No se han guardado archivos.")
import os
import io
import cv2
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.models as models
from torchvision.transforms import InterpolationMode
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_curve, auc

# ==============================================================================
# 1. CONFIGURACIÓN Y RUTAS
# ==============================================================================
DATA_PATH    = "./data/breakhis_npz"
RESULTS_PATH = "./results"
MODELS_PATH  = "./models" 

# Archivos de resultados (CSV limpio y NPY)
METRICS_FILE = os.path.join(RESULTS_PATH, "tfm_test_metrics.csv")
PREDS_FILE   = os.path.join(RESULTS_PATH, "tfm_test_predictions.npy")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Configuración visual
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
COLORS = {'Advanced': '#D62728', 'Basic': '#1F77B4', 'None': '#2CA02C'}

# Umbrales y Configuración de Modelos (Hardcoded del entrenamiento)
THRESHOLDS = {
    "None": 0.1978,
    "Basic": 0.1476,
    "Advanced": 0.1609
}

SCENARIO_CONFIGS = {
    "None":     {"head_hidden": 512, "dropout": 0.7},
    "Basic":    {"head_hidden": 128, "dropout": 0.3},
    "Advanced": {"head_hidden": 512, "dropout": 0.5}
}

# ==============================================================================
# 2. CLASES Y UTILIDADES (Grad-CAM, Dataset)
# ==============================================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x):
        self.model.zero_grad()
        output = self.model(x)
        score = output[:, 0]
        score.backward(retain_graph=True)
        
        gradients = self.gradients
        activations = self.activations
        b, k, u, v = gradients.size()
        
        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)
        
        heatmap = (weights * activations).sum(1, keepdim=True)
        heatmap = F.relu(heatmap)
        
        heatmap = heatmap.view(b, -1)
        heatmap -= heatmap.min(1, keepdim=True)[0]
        heatmap /= (heatmap.max(1, keepdim=True)[0] + 1e-8)
        heatmap = heatmap.view(b, 1, u, v)
        
        return heatmap, torch.sigmoid(output).item()

class SimpleInferenceDataset(Dataset):
    def __init__(self, file_path):
        self.data = np.load(file_path)
        # Buscar key válida (arr_0 o images)
        self.key = [k for k in self.data.files if "images" in k or "arr" in k][0]
        self.transform = T.Compose([
            T.Resize(224, interpolation=InterpolationMode.BILINEAR),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    def __len__(self): return self.data[self.key].shape[0]
    def __getitem__(self, idx):
        img_raw = self.data[self.key][idx]
        # Asegurar formato uint8 [0,255]
        if img_raw.max() <= 1.0: img_raw = (img_raw * 255).astype(np.uint8)
        else: img_raw = img_raw.astype(np.uint8)
        
        img_pil = Image.fromarray(img_raw)
        return self.transform(img_pil), img_pil

def create_model_for_inference(hparams):
    model = models.resnet50(weights=None) # No necesitamos pesos de ImageNet, cargaremos los nuestros
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, hparams["head_hidden"]),
        nn.ReLU(inplace=True),
        nn.Dropout(hparams["dropout"]),
        nn.Linear(hparams["head_hidden"], 1)
    )
    return model.to(DEVICE)

def denormalize(tensor):
    img = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)

def overlay_heatmap(img_np, heatmap_tensor):
    heatmap = heatmap_tensor.detach().squeeze().cpu().numpy()
    heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    return np.clip(0.6 * img_np + 0.4 * heatmap, 0, 1)

# ==============================================================================
# 3. FUNCIÓN: GRAFICO ROC AUC
# ==============================================================================
def plot_roc_curves():
    print("\nGenerando Gráfico ROC AUC...")
    try:
        df_metrics = pd.read_csv(
                METRICS_FILE,
                dtype={'Scenario': str},
                keep_default_na=False,
                na_filter=False
        )
        df_metrics["Scenario_norm"] = df_metrics["Scenario"].str.strip().str.lower()
        
        predictions = np.load(PREDS_FILE, allow_pickle=True).item()
        
        plt.figure(figsize=(10, 9), dpi=150)
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Azar (AUC = 0.50)')

        scenarios_order = ["None", "Basic", "Advanced"]
        
        for sc in scenarios_order:
            row = df_metrics[df_metrics["Scenario_norm"] == sc.lower()]
            if row.empty or sc not in predictions: continue
            
            auc_csv = float(row.iloc[0]["AUC"])
            y_true = predictions[sc]["y_true"]
            y_prob = predictions[sc]["y_prob"]
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            
            lw = 3.5 if sc == 'Advanced' else 2.5
            alpha = 1.0 if sc == 'Advanced' else 0.8
            
            plt.plot(fpr, tpr, color=COLORS.get(sc, 'black'), lw=lw, alpha=alpha,
                     label=f'{sc:<10} (AUC = {auc_csv:.4f})')

        plt.xlim([-0.01, 1.0])
        plt.ylim([0.0, 1.01])
        plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=13, labelpad=10)
        plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)', fontsize=13, labelpad=10)
        plt.title('Curvas ROC AUC en Conjunto de Prueba', fontsize=16, pad=20, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11, title="Escenario", title_fontsize=12, frameon=True, shadow=True)
        plt.tight_layout()
        
        out_path = os.path.join(RESULTS_PATH, "fig_roc_auc.png")
        plt.savefig(out_path, bbox_inches='tight')
        print(f"✅ Gráfico guardado en: {out_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error generando ROC: {e}")

# ==============================================================================
# 4. FUNCIÓN: GRAD-CAM (Muestras Específicas)
# ==============================================================================
def visualize_gradcam_targets(df_test, models_loaded):
    print("\n🔍 Generando visualizaciones Grad-CAM para objetivos seleccionados...")
    
    # Lista de (PatientID, índice_de_imagen) seleccionados para el TFM
    TARGETS = [
        ("14926",   2), ("16875",  16), ("21998CD",   1),
        ("21998CD",  31), ("5694",    12), ("13418DE", 1),
        ("16448",   12), ("2523", 16),
    ]
    
    for pid, idx in TARGETS:
        # 1. Localizar paciente
        row = df_test[df_test["PatientID"] == str(pid)]
        if len(row) == 0:
            print(f"Paciente {pid} no encontrado en Test Set.")
            continue
        
        row = row.iloc[0]
        fpath = os.path.join(DATA_PATH, row["FileName"])
        
        if not os.path.exists(fpath): continue
        
        # 2. Cargar imagen
        ds = SimpleInferenceDataset(fpath)
        if idx >= len(ds): continue
        
        img_tensor, _ = ds[idx]
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        img_np = denormalize(img_tensor)
        
        # 3. Plot
        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        
        # Original
        axes[0].imshow(img_np)
        axes[0].set_title(f"{row['TumorType']}\nPac: {pid}", fontsize=18, fontweight="bold")
        axes[0].axis("off")
        
        # Modelos
        for i, sc in enumerate(["None", "Basic", "Advanced"]):
            if sc not in models_loaded: continue
            
            model = models_loaded[sc]
            grad_cam = GradCAM(model, model.layer4[-1])
            heatmap, prob = grad_cam(img_tensor)
            overlay = overlay_heatmap(img_np, heatmap)
            
            axes[i+1].imshow(overlay)
            
            thresh = THRESHOLDS[sc]
            is_malign = (prob >= thresh)
            
            # Color título (Verde=Acierto, Rojo=Fallo)
            is_correct = (is_malign and row["TumorType"] == "malign") or \
                         (not is_malign and row["TumorType"] == "benign")
            color = "green" if is_correct else "red"
            
            axes[i+1].set_title(f"{sc}\nProb: {prob:.2f}", color=color, fontweight="bold", fontsize=20)
            axes[i+1].axis("off")
            
        plt.tight_layout()
        out_file = os.path.join(RESULTS_PATH, f"gradcam_{pid}_img{idx}.png")
        plt.savefig(out_file, bbox_inches='tight')
        print(f"   -> Guardado: {out_file}")
        plt.close()

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    if not os.path.exists(RESULTS_PATH): os.makedirs(RESULTS_PATH)

    # 1. GENERAR ROC AUC (Independiente de modelos cargados)
    plot_roc_curves()

    # 2. CARGAR MODELOS (Para Grad-CAM)
    print("\nCargando modelos para inferencia...")
    models_loaded = {}
    for name, cfg in SCENARIO_CONFIGS.items():
        # Ajustar nombre de archivo según lo que genera el script 2
        path_pth = os.path.join(MODELS_PATH, f"final_model_{name}.pth")
        if os.path.exists(path_pth):
            try:
                m = create_model_for_inference(cfg)
                m.load_state_dict(torch.load(path_pth, map_location=DEVICE))
                m.eval()
                models_loaded[name] = m
                print(f"{name} cargado.")
            except Exception as e: print(f"Error {name}: {e}")
        else:
            print(f"No encontrado: {path_pth}")

    if models_loaded:
        # 3. PREPARAR TEST SET (Recuperar índices exactos)
        print("\nReconstruyendo Test Set...")
        files = [f for f in os.listdir(DATA_PATH) if f.endswith(".npz")]
        df_all = pd.DataFrame([(f.split("_")[0], "_".join(f.split("_")[1:-1]), f.split("_")[-1].replace(".npz",""), f) 
                               for f in files], columns=["TumorType","Subtype","PatientID","FileName"])
        
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=3)
        _, test_idx = next(gss.split(df_all, groups=df_all["PatientID"]))
        df_test = df_all.iloc[test_idx]
        
        visualize_gradcam_targets(df_test, models_loaded)
        
        # 5. (OPCIONAL) REPORTE HTML MASIVO
        # Si quieres ejecutarlo, descomenta la siguiente línea:
        # generate_full_html_report(df_test, models_loaded) 
        
    else:
        print("No se cargaron modelos, saltando Grad-CAM.")

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import pandas as pd
from sklearn.preprocessing import StandardScaler
from PIL import Image
from transformers import AutoTokenizer

class DatasetMedicoMultimodal(Dataset):
    def __init__(self, ruta_csv="data_hospital/index_clinico.csv"):
        # 1. Leer el indexador del disco
        self.df = pd.read_csv(ruta_csv)
        
        # 2. Normalizar las columnas bioquímicas de la analítica de sangre
        self.escalador = StandardScaler()
        self.columnas_medicas = ['Glob_Blancos_uL', 'PCR_mg_L', 'Neutrofilos_porc']
        self.sangre_normalizada = self.escalador.fit_transform(self.df[self.columnas_medicas])
        
        # 3. Extraer etiquetas objetivos
        self.etiquetas = self.df['Diagnostico_Real'].values

        # 4. Inicializar Tokenizador Clínico para las notas de texto
        self.tokenizador = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        
        # Pre-tokenizar todo el texto para acelerar el bucle de entrenamiento
        self.tokens_texto = self.tokenizador(
            self.df['Nota_Clinica'].tolist(), padding=True, truncation=True, max_length=32, return_tensors="pt"
        )

        # 5. Configurar transformaciones de imágenes
        self.transformador_imagen = T.Compose([
            T.Resize((224, 224)),
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # A. Sangre
        sangre_tensor = torch.tensor(self.sangre_normalizada[idx], dtype=torch.float32)
        
        # B. Imagen (Lee de forma dinámica el archivo específico del paciente actual)
        ruta_img = self.df.iloc[idx]['Ruta_Imagen']
        img_cargada = Image.open(ruta_img)
        imagen_tensor = self.transformador_imagen(img_cargada)
        
        # C. Texto
        input_ids = self.tokens_texto['input_ids'][idx]
        attention_mask = self.tokens_texto['attention_mask'][idx]
        
        # D. Etiqueta
        etiqueta = torch.tensor(self.etiquetas[idx], dtype=torch.long)
        
        return {
            'imagen': imagen_tensor,
            'sangre': sangre_tensor,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'etiqueta': etiqueta
        }

if __name__ == "__main__":
    dataset_test = DatasetMedicoMultimodal()
    cargador_test = DataLoader(dataset_test, batch_size=4, shuffle=True)
    batch = next(iter(cargador_test))
    print("\n✔ Verificación de Escalado Exitosa:")
    print(f" -> Forma de lote de imágenes en disco: {batch['imagen'].shape}")
    print(f" -> Forma de lote de sangre normalizada: {batch['sangre'].shape}")
    print(f" -> Muestra de etiquetas reales del hospital: {batch['etiqueta']}")

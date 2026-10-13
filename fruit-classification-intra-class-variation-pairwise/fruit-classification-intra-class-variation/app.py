from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

CHECKPOINT = Path("models/siamese_fruit.pth")

class SiameseEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        input_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim),
        )
    def forward(self, image):
        return nn.functional.normalize(self.projection(self.backbone(image)), p=2, dim=1)

@st.cache_resource
def load_model(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = SiameseEncoder(payload.get("embedding_dim", 128))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload

st.set_page_config(page_title="Fruit Image Comparison", page_icon="🍌")
st.title("Fruit Classification with Intra-Class Variation")
st.subheader("Comparação de duas imagens")
st.write(
    "Envie duas imagens para comparar as características visuais aprendidas pelo modelo. "
    "O resultado depende da qualidade e representatividade do dataset."
)

if not CHECKPOINT.exists():
    st.warning(
        "Modelo treinado não encontrado. Organize um dataset rotulado e execute "
        "`python train.py --data-dir data/fruits --output models/siamese_fruit.pth`."
    )
    st.stop()

col_a, col_b = st.columns(2)
with col_a:
    file_a = st.file_uploader("Imagem A", type=["jpg", "jpeg", "png", "webp"], key="a")
with col_b:
    file_b = st.file_uploader("Imagem B", type=["jpg", "jpeg", "png", "webp"], key="b")

if file_a and file_b:
    try:
        image_a = Image.open(file_a).convert("RGB")
        image_b = Image.open(file_b).convert("RGB")
    except Exception as exc:
        st.error(f"Não foi possível abrir uma das imagens: {exc}")
        st.stop()
    with col_a:
        st.image(image_a, caption="Imagem A", use_container_width=True)
    with col_b:
        st.image(image_b, caption="Imagem B", use_container_width=True)

    size = int(load_model(str(CHECKPOINT))[1].get("image_size", 224))
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    model, payload = load_model(str(CHECKPOINT))
    tensor_a = transform(image_a).unsqueeze(0)
    tensor_b = transform(image_b).unsqueeze(0)
    with torch.inference_mode():
        emb_a = model(tensor_a)
        emb_b = model(tensor_b)
        distance = torch.linalg.vector_norm(emb_a - emb_b, dim=1).item()

    threshold = float(payload["distance_threshold"])
    st.metric("Distância entre representações", f"{distance:.4f}")
    if distance < threshold:
        st.success("O modelo estima que as imagens pertencem à mesma classe aprendida.")
    else:
        st.info("O modelo estima que as imagens pertencem a classes diferentes.")
    st.caption(
        f"Limite aprendido na validação: {threshold:.4f}. "
        "A distância não é uma probabilidade. Este protótipo não confirma a identidade biológica da fruta."
    )

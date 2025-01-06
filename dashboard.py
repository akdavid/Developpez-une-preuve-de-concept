import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from BSRGAN.models.network_rrdbnet import RRDBNet as net
from BSRGAN.utils import utils_image as util

# Chemin du modèle BSRGAN
MODEL_PATH = "BSRGAN/model_zoo/BSRGAN.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration de la page
st.set_page_config(page_title="Data Augmentation & Super Resolution", layout="wide")

# Titre
st.title("Dashboard : Augmentation de Données et Super Résolution")
st.write("Interagissez avec des images pour explorer l'augmentation de données et la super résolution.")

# Définir les transformations avancées pour augmentation de données
def get_augmentation_transforms():
    from torchvision.transforms import v2
    return v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(degrees=45),
        v2.RandomZoomOut(),
        v2.RandomPerspective(distortion_scale=0.5),
    ])

# Section 1 : Augmentation de données
st.header("Augmentation de Données")
uploaded_image = st.file_uploader("Téléversez une image pour augmentation de données", type=["jpg", "png", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Image Originale", use_container_width=False)

    num_examples = st.slider("Nombre d'exemples générés", 1, 10, 4)

    st.markdown("""
    ### Transformations utilisées :
    - **Flip Horizontal** : Retourne l'image horizontalement.
    - **Rotation Aléatoire** : Fait pivoter l'image dans une plage aléatoire de ±45 degrés.
    - **Zoom Out** : Réduit la taille de l'image dans son cadre tout en ajoutant des bordures.
    - **Perspective** : Applique une déformation en perspective pour simuler une prise de vue inclinée.
    """)

    original_size = image.size
    transformed_images = []

    for _ in range(num_examples):
        from torchvision.transforms import v2
        image_tensor = v2.ToImage()(np.array(image))
        augmented_tensor = get_augmentation_transforms()(image_tensor)
        augmented_image = to_pil_image(augmented_tensor).resize(original_size)
        transformed_images.append(augmented_image)

    st.image(transformed_images, caption=[f"Exemple {i+1}" for i in range(num_examples)], use_container_width=False)

# Section 2 : Super résolution
st.header("Super Résolution")
uploaded_image_sr = st.file_uploader("Téléversez une image pour super résolution", type=["jpg", "png", "jpeg"], key="sr")

if uploaded_image_sr:
    image_sr = Image.open(uploaded_image_sr).convert("RGB")
    st.image(image_sr, caption="Image Originale pour Super Résolution", use_container_width=False)

    # algorithm = st.selectbox("Choisissez un algorithme", ["BSRGAN"])
    algorithm = "BSRGAN"
    factor = 4  # Facteur de mise à l'échelle

    # Charger le modèle BSRGAN
    @st.cache_resource
    def load_bsrgan_model(model_path):
        """Charge le modèle BSRGAN."""
        model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=factor)
        model.load_state_dict(torch.load(model_path, weights_only=True), strict=True)
        model.eval()
        for _, param in model.named_parameters():
            param.requires_grad = False
        return model.to(DEVICE)

    # Appliquer la super résolution
    def super_resolve_bsrgan(image, model):
        """Applique BSRGAN à une image."""
        img_np = np.array(image)
        img_tensor = util.uint2tensor4(img_np).to(DEVICE)
        with torch.no_grad():
            output_tensor = model(img_tensor)
        output_np = util.tensor2uint(output_tensor)
        return Image.fromarray(output_np)

    # Exécuter la super résolution
    with st.spinner(f"Traitement avec {algorithm}..."):
        if algorithm == "BSRGAN":
            model = load_bsrgan_model(MODEL_PATH)
            sr_image = super_resolve_bsrgan(image_sr, model)

    st.image(sr_image, caption=f"Image après Super Résolution ({algorithm}, x{factor})", use_container_width=False)

    st.download_button(
        label="Télécharger l'image super-résolue",
        data=util.img2bytearray(sr_image),
        file_name=f"{algorithm}_super_resolved_x{factor}.png",
        mime="image/png"
    )

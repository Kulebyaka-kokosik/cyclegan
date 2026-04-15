from PIL import Image
import numpy as np
from cyclegan import utils, model
import streamlit as st

HORSE_VS_ZEBRA_MODEL_PATH = "./weights/horse_vs_zebra.pt"
PROVENCE_VS_MINIMAL_PATH = "./weights/kitchens.pt"
H2Z = "Horse → Zebra"
Z2H = "Zebra → Horse"
P2M = "Kitchen style: Provence → Kitchen style: Minimalism"
M2P = "Kitchen style: Minimalism → Kitchen style: Provence"



@st.cache_resource
def load_model_horse_zebra():
    cg_model = utils.load_model(HORSE_VS_ZEBRA_MODEL_PATH)
    cg_model.eval()
    return cg_model

@st.cache_resource
def load_model_kitchens():
    cg_model = utils.load_model(PROVENCE_VS_MINIMAL_PATH)
    cg_model.eval()
    return cg_model


def inference(
        model: model.CycleGAN,
        image: Image.Image,
        a2b: bool = True,
) -> np.ndarray:
    image_tensor = utils.apply_transform(image).unsqueeze(0)

    if a2b:
        output_image_tensor = model.generator_A(image_tensor).squeeze(0)
    else:
        output_image_tensor = model.generator_B(image_tensor).squeeze(0)

    output_image = utils.denormalize(output_image_tensor)

    return output_image


def resize_max_side(image: Image.Image, max_size=256) -> Image.Image:
    w, h = image.size

    scale = max_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return image.resize((new_w, new_h), Image.Resampling.BICUBIC)



def main():

    st.title("Domains transfer")

    # hors_zebra_model = load_model_horse_zebra()
    kitchen_model = load_model_kitchens()

    direction = st.selectbox(
        "Direction",
        [P2M, M2P]
    )

    a2b = True
    if direction == Z2H or direction == M2P:
        a2b = False

    # cg_model = hors_zebra_model if direction == H2Z or direction == Z2H else kitchen_model
    cg_model = kitchen_model

    uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image = resize_max_side(image)

        output_image = inference(cg_model, image, a2b=a2b)

        col1, col2 = st.columns(2)

        col1.image(image, caption="Input")
        col2.image(output_image, caption="Output")


main()

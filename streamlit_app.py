from PIL import Image
import numpy as np
from cyclegan import utils, model
import streamlit as st

PROVENCE_VS_MINIMAL_PATH = "./weights/kitchens.pt"
P2M = "Kitchen style: Provence → Kitchen style: Minimalism"
M2P = "Kitchen style: Minimalism → Kitchen style: Provence"


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

    kitchen_model = load_model_kitchens()

    direction = st.selectbox(
        "Direction",
        [P2M, M2P]
    )

    a2b = True
    example_image_path = "./assets/provence.jpg"
    if direction == M2P:
        a2b = False
        example_image_path = "./assets/minimalism.jpg"

    if st.session_state.get("direction") != direction:
        st.session_state["direction"] = direction
        st.session_state["use_example"] = False

    cg_model = kitchen_model

    st.image(example_image_path, caption="Example image", width=200)
    use_example = st.button("Use the example")

    if use_example:
        st.session_state["use_example"] = True

    uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

    if uploaded_file and not use_example:
        st.session_state["use_example"] = False

    if uploaded_file or st.session_state.get("use_example"):
        if st.session_state.get("use_example"):
            image = Image.open(example_image_path).convert("RGB")
        else:
            image = Image.open(uploaded_file).convert("RGB")

        image = resize_max_side(image)

        output_image = inference(cg_model, image, a2b=a2b)

        col1, col2 = st.columns(2)

        col1.image(image, caption="Input")
        col2.image(output_image, caption="Output")


main()

import gradio as gr
import model
from config import app_config


def init():
    if model != None:
        print("Initializing App...")
    app_config.model = model.load_model()


def clear():
    return None, 2, None, None, None


def create_interface():
    md = """
       # Famous Landmark Classifier using CNN
       **Choose an image containing any of the `50 possible classes` of world famous landmarks,** 
       **choose the number of prediction required (k) and hit `Predict`, model will try to identify**
       **the landmark in the image.** 
       **Please note that the model is trained on a small set of only 4,000 images hence it may not** 
       **be right all the time, but its fun to try out.**  
       Visit the [project's repo](https://github.com/sssingh/landmark-classification-tagging)

       ***Please be patient after clicking on example images, they are loaded from Git Large File System (LFS) and first time it may take few seconds to load***
       """
    with gr.Blocks(
        title=app_config.title, theme=app_config.theme, css=app_config.css
    ) as app:
        with gr.Row():
            gr.Markdown(md)
            with gr.Accordion(
                "Expand to see 50 classes:", open=False, elem_classes="accordion"
            ):
                gr.JSON(app_config.classes, elem_classes="json-box")
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="pil", elem_classes="image-picker")
                k = gr.Slider(
                    label="Number of predictions (k):",
                    minimum=2,
                    maximum=5,
                    value=2,
                    step=1,
                    elem_classes="slider",
                )
                with gr.Row():
                    submit_btn = gr.Button(
                        "Predict",
                        icon="assets/button-icon.png",
                        elem_classes="submit-button",
                    )
                    clear_btn = gr.ClearButton(elem_classes="clear-button")
            with gr.Column():
                landmarks = gr.JSON(
                    label="Predicted Landmarks:", elem_classes="json-box"
                )
                proba = gr.JSON(
                    label="Predicted Probabilities:", elem_classes="json-box"
                )
                plot = gr.Plot(container=True, elem_classes="plot")
        with gr.Row():
            with gr.Accordion(
                "Expand for examples:", open=False, elem_classes="accordion"
            ):
                gr.Examples(
                    examples=[
                        ["assets/examples/gateway-of-india.jpg", 3],
                        ["assets/examples/grand-canyon.jpg", 2],
                        ["assets/examples/opera-house.jpg", 3],
                        ["assets/examples/stone-henge.jpg", 4],
                        ["assets/examples/temple-of-zeus.jpg", 5],
                    ],
                    inputs=[img, k],
                    outputs=[landmarks, proba],
                    elem_id="examples",
                )
        submit_btn.click(
            fn=model.predict, inputs=[img, k], outputs=[landmarks, proba, plot]
        )
        clear_btn.click(fn=clear, inputs=[], outputs=[img, k, landmarks, proba, plot])
        img.clear(fn=clear, inputs=[], outputs=[img, k, landmarks, proba, plot])
    return app


if __name__ == "__main__":
    init()
    app = create_interface()
    app.launch()

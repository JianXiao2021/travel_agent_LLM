from graph.graph import init_app
import os
from dotenv import load_dotenv, find_dotenv

if __name__ == "__main__":
    _ = load_dotenv(find_dotenv())
    app = init_app(model_name="gpt-3.5-turbo")
    image_path = "graph.png"
    graph_image = app.get_graph().draw_mermaid_png()
    with open(image_path, "wb") as img_file:
        img_file.write(graph_image)
    os.startfile(image_path)
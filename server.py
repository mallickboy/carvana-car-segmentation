from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import io
import base64  
from modules.cnn import model
from PIL import Image


MODEL= model(path= "./modules/model.keras")

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def upload_form():
    html_content = """
    <html>
        <body>
            <h2>Upload an Image</h2>
            <form action="/upload/" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required>
                <input type="submit" value="Upload Image">
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Route to handle image upload and processing
@app.post("/upload/", response_class=HTMLResponse)
async def create_upload_file(file: UploadFile = File(...)):
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes))   # reading original

    # inverted_image = MODEL.invert_image(image)        # process predict
    org_resized, pred = MODEL.predict_image(image)        # process predict

    original_img_byte_arr = io.BytesIO()
    output_img_byte_arr = io.BytesIO()

    org_resized.save(original_img_byte_arr, format="PNG")
    pred.save(output_img_byte_arr, format="PNG")

    original_img_byte_arr.seek(0)
    output_img_byte_arr.seek(0)

    # generate HTML content
    original_base64 = base64.b64encode(original_img_byte_arr.getvalue()).decode('utf-8')
    inverted_base64 = base64.b64encode(output_img_byte_arr.getvalue()).decode('utf-8')

    html_content = f"""
    <html>
        <body>
            <h2>Upload a New Image</h2>
            <form action="/upload/" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required>
                <input type="submit" value="Upload Image">
            </form>
            <div style="display: flex; flex-wrap: wrap; gap: 20px;">
                <div style="text-align: center; flex: 1 1 200px;">
                    <h2>Original Image</h2>
                    <img src="data:image/png;base64,{original_base64}" alt="Original Image" style="height: 50vh; max-width: 100%; object-fit: contain;">
                </div>
                <div style="text-align: center; flex: 1 1 200px;">
                    <h2>Prediction</h2>
                    <img src="data:image/png;base64,{inverted_base64}" alt="Inverted Image" style="height: 50vh; max-width: 100%; object-fit: contain;">
                </div>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

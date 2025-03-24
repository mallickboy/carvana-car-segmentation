<h1 align="center"> Car Segmentation (Carvana Dataset)</h1>
Colab Notebook on car segmentation of Carvana dataset as a task of BigVision

## DOCS
- Google Colab Notebook:  https://colab.research.google.com/drive/1ElMK4pRj4wyJqiaHRGS56WiKx7oguEjW

- Report: https://drive.google.com/file/d/1bGEoxOrEH70-53lve-tKV2YkJybuM3rM/view

## Server Deployment Setup

#### Pull code
```
git remote add car https://github.com/mallickboy/carvana-car-segmentation.git

git pull car main
```

#### Virtual environment Setup
```
py -3.11 -m venv .venv  

source .venv/Scripts/activate  # For Windows  

source .venv/bin/activate  # For Linux/macOS  

pip install -r requirements.txt  
```
#### Run server
```
uvicorn server:app --host 0.0.0.0 --port 8000  

http://127.0.0.1:8000 (or your_server_ip:8000)
```
#### Deactivate Virtual Environment

```
deactivate  
```

## Output

The model performs well on images with simple backgrounds.

![Image](https://github.com/user-attachments/assets/1ec82d7a-bd3d-42bb-bd9d-ce16990bce5a)

![Image](https://github.com/user-attachments/assets/64def700-d435-4e39-ada8-54f090244521)

![Image](https://github.com/user-attachments/assets/3213ee0f-7980-45e3-9d53-62ea1a77478d)

# carvana-car-segmentation
Colab Notebook on car segmentation of Carvana dataset as a task of BigVision

# Deployment setup

#### Pull code
```
git remote add car https://github.com/mallickboy/carvana-car-segmentation.git

git pull car main
```

#### Virtual environment Setup
```
py -3.11 -m venv .venv

.\.venv\Scripts\activate

pip install -r requirements.txt
```
#### Run server
```
uvicorn server:app

http://127.0.0.1:8000
```
#### Deactivate Virtual Environment

```

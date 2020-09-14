# food-identification
An artificial intelligence application to identify food. Using flutter, python, and tensorflow. 

Steps
--------
Build Docker Image according to Dockerfile named "ai-food"
```
docker build -t ai-food .
```

Create docker container with the image you just created
```
docker run --name ai-food -dp 5000:5000 ai-food
```

CURL to check it works
```
curl localhost:5000
```

Enjoy using <b>Postman</b> to test
```
POST to /upload
form-data:
  key: file
  value: your file <Click and select it>
```
if stuck
docker stop ai-food 
docker rm ai-food 
GOOD LUCK
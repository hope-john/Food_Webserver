docker stop ai-food
docker rm ai-food
docker build -t ai-food .
docker run --name ai-food -dp 5000:5000 ai-food
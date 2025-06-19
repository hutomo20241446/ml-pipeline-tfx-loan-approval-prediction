# Gunakan TensorFlow Serving sebagai base image
FROM tensorflow/serving:latest

# Tentukan direktori kerja dalam container
WORKDIR /app

# Salin model yang telah dipush ke dalam container
COPY ./serving_model /models

# Tentukan environment variable untuk TensorFlow Serving
ENV MODEL_NAME=loan_model
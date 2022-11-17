FROM python:3.9

WORKDIR /quickdraws

COPY requirements.txt .

COPY ./src ./src

COPY ./example ./example

COPY run_example.sh .

RUN pip install -r requirements.txt 

RUN chmod +x src/RHEmcmt

RUN mkdir output/

CMD ["bash", "./run_example.sh"]

#Build an image: docker build -t python-imdb .
#Run the container: docker run --gpus all python-imdb   e
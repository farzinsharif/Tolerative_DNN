FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime


WORKDIR /workspace

COPY environment.yml .


RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN conda env update -n base -f environment.yml && conda clean -afy

COPY TMR_in_CNN.py .

CMD ["python", "TMR_in_CNN.py"]

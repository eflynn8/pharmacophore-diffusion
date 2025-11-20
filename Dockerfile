FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install Miniconda
RUN apt-get update && apt-get install -y wget git && rm -rf /var/lib/apt/lists/*
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r


RUN conda install conda-libmamba-solver && conda config --set solver libmamba

# Copy environment file and install dependencies
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy
ENV CONDA_DEFAULT_ENV=forge-env
RUN conda init
RUN echo "conda activate forge-env" >> ~/.bashrc

# Clone repository and setup workspace
RUN git clone https://github.com/eflynn8/pharmacophore-diffusion /app
WORKDIR /app

# Copy the Streamlit app
COPY app.py .

# Download model weights and config
RUN mkdir -p model_dir/checkpoints
RUN wget -O model_dir/checkpoints/last.ckpt https://bits.csb.pitt.edu/files/pharmacoforge/last.ckpt
RUN wget -O model_dir/config.yaml https://bits.csb.pitt.edu/files/pharmacoforge/config.yaml

# Expose port and set entrypoint
EXPOSE 8080
CMD ["conda", "run", "-n", "forge-env", "streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
FROM --platform=linux/amd64  pytorch/pytorch

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1

RUN adduser --system --group user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt opt/app

RUN python -m pip install \
    --no-color \
    --requirement requirements.txt

COPY --chown=user:user helper.py /opt/app
COPY --chown=user:user inference.py /opt/app
COPY --chown=user:user gadnet.py /opt/app
COPY --chown=user:user checkpoints /opt/app/checkpoints
# Create the output directory
# COPY --chown=user:user output /app/output

# WORKDIR ..

ENTRYPOINT ["python", "inference.py"]

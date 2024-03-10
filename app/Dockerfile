FROM --platform=linux/amd64  pytorch/pytorch

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1

RUN adduser --system --group user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app

RUN python -m pip install \
    --no-color \
    --requirement requirements.txt

COPY --chown=user:user helper.py /opt/app
COPY --chown=user:user inference.py /opt/app
COPY --chown=user:user gadnet.py /opt/app
COPY --chown=user:user checkpoints /opt/app/checkpoints
# # Create the output directory and set appropriate permissions
# USER root
# RUN mkdir /opt/app/output && chown user:user /opt/app/output
# USER user

# WORKDIR ..

ENTRYPOINT ["python", "inference.py"]

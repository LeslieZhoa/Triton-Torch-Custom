ARG BASE_IMAGE

FROM ${BASE_IMAGE}

COPY models /models

COPY libfusebias.so /

ENV LD_LIBRARY_PATH="/opt/tritonserver/backends/pytorch:/opt/tritonserver/lib/pytorch:${LD_LIBRARY_PATH}"
ENV LD_PRELOAD="/libfusebias.so"
EXPOSE 8001

ENTRYPOINT ["tritonserver", "--model-repository=/models"]

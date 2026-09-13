FROM eclipse-temurin:21-jdk

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates curl maven \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work
ENV HOME=/tmp

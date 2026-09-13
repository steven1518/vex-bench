FROM vex-bench-python:latest

ARG OPENCODE_VERSION=1.15.5

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/* \
 && npm install -g opencode-ai@${OPENCODE_VERSION}

ENTRYPOINT ["opencode"]

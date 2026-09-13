FROM vex-bench-python:latest

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/* \
 && npm install -g @openai/codex@0.131.0

ENTRYPOINT ["codex"]

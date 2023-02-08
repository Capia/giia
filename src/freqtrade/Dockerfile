# If this version is updated, ensure that the `requirements.txt` python version is updated, as well as your local
# python version
FROM freqtradeorg/freqtrade:2023.1

# Switch user to root if you must install something from apt. Don't forget to switch the user back below!
USER root

RUN apt install -y curl

USER ftuser

# It is important to note that we reflect the directory structure of the repo. This is to ensure dependencies/imports
# and configs are aligned
ENV PYTHONPATH="/capia/src"
COPY --chown=ftuser src/ /capia/src
WORKDIR /capia/src/freqtrade

RUN pip install --user tqdm

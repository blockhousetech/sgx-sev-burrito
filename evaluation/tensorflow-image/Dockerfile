##
## Burrito
## Copyright (C) 2023 The Blockhouse Technology Limited (TBTL)
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU Affero General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Affero General Public License for more details.
##
## You should have received a copy of the GNU Affero General Public License
## along with this program. If not, see <https://www.gnu.org/licenses/>.
##
FROM tensorflow/tensorflow:2.9.1

RUN apt update
RUN apt install -y openssh-client

ARG PYTHON_SCRIPT
ENV PYTHON_SCRIPT=${PYTHON_SCRIPT}

COPY ${PYTHON_SCRIPT} .
COPY go.sh .

# Files related to report generation
RUN mkdir -p ../trusted_owner/App/proto/gen
COPY report_generation/sev-report-generation.py .
COPY report_generation/.env .
COPY report_generation/server_pb2.py ../trusted_owner/App/proto/gen/
COPY report_generation/server_pb2_grpc.py ../trusted_owner/App/proto/gen/

RUN chmod +x go.sh
RUN chmod +x ${PYTHON_SCRIPT}

RUN mkdir -p /root/.ssh
COPY throwaway /root/.ssh/id_rsa
RUN chmod 600 /root/.ssh/id_rsa
RUN echo "Host *\n\tStrictHostKeyChecking no\n" >> /root/.ssh/config

RUN pip install python-decouple

# word2vec.py
RUN mkdir -p /secrets
RUN pip install tqdm

# text_classification.py

RUN pip install matplotlib

# preprocessing_layers.py

RUN pip install pandas
RUN apt update
RUN apt install -y graphviz
RUN pip install pydot

# segmentation.py

RUN apt install -y git
RUN pip install git+https://github.com/tensorflow/examples.git

# music_generation.py

RUN apt install -y fluidsynth
RUN pip install --upgrade pyfluidsynth
RUN pip install pretty_midi

# imbalanced_data.py

RUN pip install seaborn
RUN pip install scikit-learn 

ENTRYPOINT ./go.sh ${PYTHON_SCRIPT}

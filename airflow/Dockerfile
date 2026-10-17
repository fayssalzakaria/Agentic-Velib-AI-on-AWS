# Image Airflow 3 + environnement Python isolé pour le pipeline et dbt.
# Le pipeline (pandas 3, deltalake, dbt) vit dans /opt/velib-venv pour ne pas
# entrer en conflit avec les dépendances d'Airflow.
FROM apache/airflow:3.3.2-python3.12

USER root
RUN python -m venv /opt/velib-venv \
 && mkdir -p /opt/airflow/dbt_state /home/airflow/.cache/huggingface \
 && chown -R airflow: /opt/velib-venv /opt/airflow/dbt_state /home/airflow/.cache

USER airflow
COPY --chown=airflow pipeline/requirements.txt /tmp/pipeline-requirements.txt
# pendulum : Airflow le cherche dans le venv avant chaque tâche external_python
RUN /opt/velib-venv/bin/pip install --no-cache-dir -r /tmp/pipeline-requirements.txt pendulum
COPY --chown=airflow pipeline/requirements-embed.txt /tmp/pipeline-requirements-embed.txt
RUN /opt/velib-venv/bin/pip install --no-cache-dir -r /tmp/pipeline-requirements-embed.txt
RUN pip install --no-cache-dir "apache-airflow==3.3.2" "astronomer-cosmos==1.15.1"

FROM python:3
COPY . /app
WORKDIR /app
RUN ls
RUN pip install pip-review
RUN pip-review --auto
RUN pip install -r requirements.txt
CMD sh ./run.sh

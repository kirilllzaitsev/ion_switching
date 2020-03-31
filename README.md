## University of Liverpool - Ion Switching challenge on Kaggle https://www.kaggle.com/c/liverpool-ion-switching/data
### Clone

- Clone this repo to your local machine using `https://github.com/cyberpunk317/ion_switching.git`

### Setup and run

- Firstly, install the required packages

```shell
$ pip install -r requirements.txt
```

- Now run bash-script that will launch training process

```shell
$ sh run.sh
```
### Inference
After the training is complete, model parameters are loaded into models/ folder.
To examine actual predictions, run in shell:
```shell
$ cd models/
$ cat submission.csv
```

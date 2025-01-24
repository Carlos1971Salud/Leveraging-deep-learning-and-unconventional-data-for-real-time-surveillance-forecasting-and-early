# A Deep-Learning Model for Building an Early Warning System for Respiratory Infections

This repository includes datasets and codes for building 3 different Early Warning Systems (EWS):
- An EWS for predicting COVID-19 outbreaks in different provinces of Canada.
- An EWS for predicting influenza outbreaks in different provinces of Canada.
- An EWS for predicting COVID-19 outbreaks for southern African countries.

Accordingly, there are three folders in the "Datasets" folder:
- **COVID19_Canada:** includes four datasets, namely, daily COVID-19 number of cases, Google trends on COIVD-19 topic, Air Quality data (i.e. CO, NO2, SO2, O3), and Reddit posts in different provinces of Canada. All the datasets cover the period between May 1st, 2020 to July 31st, 2022 on daily basis.
- **Influenza_Canada:** includes three datasets, namely, weekly influenza number of cases, Google trends on influenza topic, and weather data (i.e. minimum temperature, total rain, total precipitation) in different provinces of Canada. All the datasets cover the period between september 1st, 2013 to September 1st, 2023, on weekly basis.
- **COVID19_Africa:** includes five datasets, namely, daily COVID-19 number of cases, Google trends on COVID-19 topic, Wiki Trends, Google News data, and air quality (i.e. CO, NO2, SO2, O3, UV-index) data for different African countries. All the datasets cover the period between June 1st, 2020 to July 31st, 2022, on daily basis.

Moreover, there are three python files:
- **CA_COVID19.py:** consumes the datasets available in the "COVID19_Canada" folder to build a deep-learning model for forecasting COVID-19 outbreaks in different provinces of Canada.
- **CA_Influenza.py:** consumes the datasets available in the "Influenza_Canada" folder to build a deep-learning model for forecasting influenza outbreaks in different provinces of Canada.
- **Africa_COVID19.py:** consumes the datasets available in the "COVID19_Africa" folder to build a deep-learning model for forecasting COVID-19 outbreaks in southern African countries.

To run any of these files, the following packages need to be installed:
```sh
torch-geometric >= 2.6.1
torch-gemetric-temporal >= 0.54
SciPy >= 1.15.1
Scikit-learn >= 1.0
```

In the "Transformers" folder, there are six different python files that trian transformer-based models, namely, [Informer](https://huggingface.co/docs/transformers/en/model_doc/informer) and [Autoformer](https://huggingface.co/docs/transformers/en/model_doc/autoformer) to build a forecasting model:
- **Informer_CA_COVID19.py:** consumes the datasets available in the "COVID19_Canada" folder to train an Informer model for forecasting COVID-19 outbreaks in Ontario.
- **Autoformer_CA_COVID19.py:** consumes the datases available in the "COVID19_Canada" folder to train an Autoformer model for forecasting COVID-19 outbreaks in Ontario.
- **Informer_CA_Influenza.py:** consumes the datasets available in the "Influenza_Canada" folder to train an Informer model for forecasting influenza outbreaks in Quebec.
- **Autoformer_CA_Influenza.py:** consumes the datasets available in the "Influenza_Canada" folder to train an Autoformer model for forecasting influenza outbreaks in Quebec.
- **Informer_Africa_COVID19.py:** consumes the datasets available in the "COVID19_Africa" folder to train an Informer model for forecasting COVID-19 outbreaks in South Africa.
- **Autoformer_Africa_COVID19.py:** consumes the datasets available in the "COVID19_Africa" folder to train an Autoformer model for forecasting COVID-19 outbreaks in South Africa.

To run any of the above files, the following packages need to be installed:
```sh
torch >= 2.5
torchmetrics >= 1.6.1
transformers >= 4.48
SciPy >= 1.15.1
Scikit-learn >= 1.0
```

To run any python file, after installing the required libraries, simply download the package and run the files.

For more information on the models or the datasets, Please refer to our manuscript below. Moreover, if you use any of our data or code in your work, please kindly cite our manuscript below:

Z Movahedi Nia, L Seyyed-Kalantari, M Goitom, B Mellado, A Ahmadi, A Asgary, et al, Leveraging Deep-Learning and Unconventional Data for Real-Time Surveillance, Forecasting, and Early Warning of Respiratory Pathogens Outbreak, Artificial Intelligence in Medicine.

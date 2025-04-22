# RingTool
## Description

**Prevent pushing pyc files into Git**.
```sh
pip install pre-commit
pre-commit install
```


RingTool is an open platform for health sensing and data analysis with smart rings. It provides a set of tools and libraries for developers to create applications that can interact with smart rings, collect data, and perform analysis on that data. The platform is designed to be flexible and extensible, allowing developers to build custom solutions for a wide range of health and wellness applications.

## Features
- **Data Collection**: RingTool provides APIs and libraries for collecting data from smart rings, including raw 3-Channel PPG, 6-axis IMU, and 3-point Temprature sensors.
- **Data Analysis**: The platform includes tools for analyzing the collected data, calculating heart rate (HR), heart rate variability (HRV), respiratory rate (RR), blood oxygen saturation (SpO2), and blood pressure (BP).

## Usage
1. **Installation**: To use RingTool, you need to install the required libraries and dependencies. You can do this by running the following command:
   ```bash
   pip install -r requirements.txt
   ```
2. **Data Collection**: Put your data under the `data` folder. The data should be npy format. 
``` 
data_daily.npy (data_sport.npydata_health.npy) 
- subject
  ring1: timestamp,green,ir,red,ax,ay,az
  ring2: timestamp,green,ir,red,ax,ay,az
  bvp: timestamp,bvp
  hr: timestamp,hr
  spo2: timestamp,spo2
  resp: timestamp,resp
  ecg: timestamp,ecg
  ecg_hr: timestamp,ecg_hr
  ecg_rr: timestamp,ecg_rr
  samsung: timestamp,hr
  oura: start, end, hr
  BP: start, end,sys,dia
  Experiment: Health, Daily, Sport
  Labels: start, end, label
```
3. **Configuration**: Configure the parameters for data collection and analysis in the `config` folder. You can specify the Health metrics, and activity settings.
4. **Train**: TODO
5. **Evaluate**: TODO

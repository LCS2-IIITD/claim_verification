# Accenture_LCS2_Project

Currently the app supports claims related to medical domain like:
- Effectiveness of masks on COVID-19.
- Does COVID-19 cause pnuemonia?
- Are vaccines effective?

  
Steps to run the app
- Clone the Repo.
- Create a new env using ```conda create -n env python=3.8``` and activate this env using ```conda activate env```
- Run ```conda install -c conda-forge openjdk=11```.
- Run ```pip install -r requirements.txt```.
- Run ```conda install -c conda-forge pyjnius```.
- Run ```conda install -c pytorch faiss-gpu```.
- Run ```echo $JAVA_HOME``` and set ```export JAVA_HOME=path/to/java```.
- Run ```sh cord19_index.sh``` to download and extract the lucene-indexed version of CORD-19 data.
- Now run ```nohup streamlit run new.py```.
- Open https://localhost:8501 to view the app.

# stonks
 
 To run this code, clone the repo and start by creating a `.env` folder at the root of the project.
 You'll need to put in your reddit specific PRAW credentials using the following format:
 
 ```text
client_id=<your-client-id>
client_secret=<your-client-secret>
user_agent=<your-reddit-username>
 ```
 
 Next, create a local virtual environment using conda:
 
 install env
 ```shell script
conda env create -f environment.yaml
```

to update env after adding new source/dependencies:
```shell script
conda env update -f environment.yaml
```

Once you have activated your conda virtual environment on your machine, at the project root, generate datasets locally, run the following:
```shell script
python main.py
```

This can potentially take a long time to run (and will depend on the global vars, `STOCKS` abd 'AMOUNT' in `reddit_data.py`.  

Now, you should have a `training_data.csv` with cleaned data ready to be used by an RNN or LSTM deep learning model for stock predictions.

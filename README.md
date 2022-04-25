# stonks

For more information related to our experiment methodology, please see the PDF report at the root.
 
To run this code, clone the repo and start by creating a `.env` folder at the root of the project.
 
You'll need to put in your reddit specific PRAW credentials using the following format, as well as comet_ml credentials if you want to log your experiments.  If you want to log your experiments, `lstm.py` is set up to log.  If you don't want to log your experiments, run `lstm_no_log.py`:
 
 ```text
client_id=<your-client-id>
client_secret=<your-client-secret>
user_agent=<your-reddit-username>

comet_api_key=<your-api-key>
comet_project_name=<your-project-name>
comet_workspace=<your-workspace-name>
 ```
 
Next, create a local virtual environment using conda:
 
 install env
 ```shell script
conda env create -f environment.yaml
```

To update env after adding new source/dependencies:
```shell script
conda env update -f environment.yaml
```

Once you have activated your conda virtual environment on your machine, at the project root, generate datasets locally, run the following:
```shell script
python main.py
```

Ensure everything in `main.py` is uncommented in order to do a scrape of reddit, yahoo finance, data clean and join, and model train.  By default, it will train on a subset of data at the daily granularity with aggregated sentiment scores by day.  To specify a different type of join (like comment per row granularity, with backfilled pricing data per comment), you'll need to edit global variables in `get_data_scripts/training_data.py` as well as `lstm.py` or `lstm_no_logging.py`. 

The global vars in `main.py` specify which subreddit to scrape, how many recent posts to include, and which stock to evaluate in that subreddit.  This code is only made to work with investing related subreddits on Reddit.com and has only been formally evaluated with the subreddit, "wallstreetbets", as it has the highest comment volume.

This can potentially take a long time to run (and will depend on the global vars, `STOCKS` and `AMOUNT` in `reddit_data.py`.  

Now, you should have a `training_data.csv` with cleaned data ready to be used by an RNN or LSTM deep learning model for stock predictions.

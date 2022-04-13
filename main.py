from get_data_scripts.reddit_data import get_post_statistics

if __name__ == "__main__":
    
    # saves reddit posts and comments into datasets
    get_post_statistics("wallstreetbets")
    get_post_statistics("investing")

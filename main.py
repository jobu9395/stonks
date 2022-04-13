from get_data_scripts.reddit_data import get_post_statistics


def main():
    get_post_statistics("wallstreetbets")
    get_post_statistics("investing")

if __name__ == "__main__":
    main()
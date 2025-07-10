import os
import re
import json
import requests
import time
from config.logger import CustomLogger


class WebScraperMarineTraffic:
    """
    A web scraper for MarineTraffic news articles using the GraphQL API.

    This class handles fetching, standardizing, and storing news articles.
    It interacts with the MarineTraffic GraphQL endpoint to retrieve articles,
    standardizes the data into a consistent format, and saves the articles in a JSON file.

    Attributes:
        logger (CustomLogger): Logger instance for tracking operations.
        _session (requests.Session): Session object configured for HTTP requests.
        dataset_dir (str): Directory path where scraped data will be stored.
        data_file (str): Full file path for the JSON file containing articles.
        graphql_url (str): URL of the MarineTraffic GraphQL API endpoint.
    """

    def __init__(self):
        """
        Initialize the WebScraperMarineTraffic instance.

        Sets up logging, creates an HTTP session with proper headers,
        determines the dataset storage directory, ensures it exists,
        and defines the path to the JSON data file.
        """
        self.logger = CustomLogger(name="WebScraperMarineTraffic")
        self.logger.ok("WebScraperMarineTraffic initialized")
        self._session = self._create_session()
        self.dataset_dir = self._get_dataset_directory()
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.data_file = os.path.join(self.dataset_dir, "marinetraffic_articles.json")
        self.graphql_url = "https://news.marinetraffic.com/graphql"

    def _create_session(self) -> requests.Session:
        """
        Create and configure an HTTP session for web requests.

        Returns:
            requests.Session: A session with preset headers including
                              Content-Type and a custom User-Agent.
        """
        session = requests.Session()
        session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/115.0 Safari/537.36"
            )
        })
        return session

    def _get_dataset_directory(self) -> str:
        """
        Determine the dataset directory where articles will be stored.

        This method checks for the existence of a 'datasets_processed' directory
        in the current or parent directory and constructs the full path for
        storing MarineTraffic news articles.

        Returns:
            str: The full directory path for storing the dataset.
        """
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            current_dir = os.getcwd()
        if os.path.exists(os.path.join(current_dir, "datasets_processed")):
            base_dir = current_dir
        elif os.path.exists(os.path.join(current_dir, "..", "datasets_processed")):
            base_dir = os.path.abspath(os.path.join(current_dir, ".."))
        else:
            base_dir = current_dir
        dataset_dir = os.path.join(base_dir, "datasets_processed", "newsarticles", "marinetraffic")
        return dataset_dir

    def _standardize_text(self, text: str) -> str:
        """
        Standardize text by replacing multiple whitespaces with a single space and stripping it.

        Args:
            text (str): The original text string.

        Returns:
            str: The cleaned and standardized text.
        """
        return re.sub(r'\s+', ' ', text).strip()

    def _load_existing_articles(self) -> list:
        """
        Load already scraped articles from the JSON file.

        Returns:
            list: A list of previously saved articles. If the file doesn't exist
                  or an error occurs during loading, returns an empty list.
        """
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, "r", encoding="utf-8") as f:
                    articles = json.load(f)
                return articles
            except Exception as e:
                self.logger.error(f"Error loading existing articles: {e}")
                return []
        return []

    def save_articles_to_json(self, articles):
        """
        Save the provided list of articles to a JSON file.

        Args:
            articles (list): A list of article dictionaries to be saved.

        Logs:
            Confirmation message on success, or an error message if saving fails.
        """
        try:
            with open(self.data_file, "w", encoding="utf-8") as f:
                json.dump(articles, f, ensure_ascii=False, indent=4)
            self.logger.ok(f"Articles saved to {self.data_file}")
        except Exception as e:
            self.logger.error(f"Error saving articles to JSON: {e}")

    def _fetch_articles_graphql(self) -> list:
        """
        Fetch articles using the GraphQL API from MarineTraffic.

        Sends a POST request with a predefined query and variables to retrieve the latest articles
        by specified categories. It then flattens the returned nested list of articles.

        Returns:
            list: A list of article dictionaries fetched from the API. Returns an empty list if an error occurs.
        """
        query = """
        query latestArticlesByCategories(
            $categoryIds: [ID!]!
            $excludedArticleIds: [ID!]!
            $limitPerCategory: Int!
          ) {
            latestArticlesByCategories(
              categoryIds: $categoryIds
              excludedArticleIds: $excludedArticleIds
              limitPerCategory: $limitPerCategory
            ) {
              id
              title
              content
              publishedAt
              slug
              category {
                data {
                  id
                  attributes {
                    name
                  }
                }
              }
              author {
                data {
                  attributes {
                    name
                    image {
                      data {
                        attributes {
                          url
                        }
                      }
                    }
                  }
                }
              }
              assets {
                data {
                  id
                  attributes {
                    assetId
                    assetName
                    assetType
                  }
                }
              }
              media {
                data {
                  attributes {
                    url
                  }
                }
              }
            }
          }
        """

        variables = {
            "categoryIds": ["14", "16", "17", "23", "26", "34"],
            "excludedArticleIds": [],
            "limitPerCategory": 100,
        }
        payload = {
            "operationName": "latestArticlesByCategories",
            "query": query,
            "variables": variables
        }
        try:
            response = self._session.post(self.graphql_url, json=payload)
            response.raise_for_status()
            result = response.json()
            articles_nested = result.get("data", {}).get("latestArticlesByCategories", [])
            # Flatten the nested list (each sublist corresponds to a category)
            articles = []
            for sublist in articles_nested:
                if isinstance(sublist, list):
                    articles.extend(sublist)
            return articles
        except Exception as e:
            self.logger.error(f"Error fetching GraphQL data: {e}")
            return []

    def _standardize_article(self, article: dict) -> dict:
        """
        Standardize a raw article into a uniform dictionary format.

        Constructs a URL for the article in the format:
            https://news.marinetraffic.com/en/maritime-news/<category_id>/<article_id>/<slug>

        Args:
            article (dict): A dictionary representing a raw article fetched from the API.

        Returns:
            dict: A standardized dictionary with keys 'url', 'headline', 'publication_date',
                  'author', and 'content'.
        """
        category_data = article.get("category", {}).get("data", {})
        category_id = category_data.get("id", "")
        article_id = article.get("id", "")
        slug = article.get("slug", "")
        url = f"https://news.marinetraffic.com/en/maritime-news/{category_id}/{article_id}/{slug}"
        headline = article.get("title", "")
        publication_date = article.get("publishedAt", "")

        author = ""
        author_data = article.get("author", {}).get("data", {})
        if author_data:
            author = author_data.get("attributes", {}).get("name", "")
        content = self._standardize_text(article.get("content", ""))
        return {
            "url": url,
            "headline": headline,
            "publication_date": publication_date,
            "author": author,
            "content": content
        }

    def scrape_marinetraffic(self) -> list:
        """
        Scrape MarineTraffic articles via the GraphQL API.

        This method fetches new articles, loads existing articles to avoid duplicates,
        standardizes new articles, and saves the combined list to a JSON file.

        Returns:
            list: A combined list of both existing and newly scraped articles.
        """
        self.logger.info("Fetching articles via GraphQL")
        articles_raw = self._fetch_articles_graphql()
        self.logger.info(f"Fetched {len(articles_raw)} articles from GraphQL endpoint")

        existing_articles = self._load_existing_articles()
        existing_urls = {article.get("url") for article in existing_articles}

        new_articles = []
        for article in articles_raw:
            standardized = self._standardize_article(article)
            if standardized["url"] not in existing_urls:
                new_articles.append(standardized)
            else:
                pass

        self.logger.info(f"New articles to scrape: {len(new_articles)}")

        all_articles = existing_articles + new_articles
        self.save_articles_to_json(all_articles)
        return all_articles
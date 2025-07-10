import os
import re
import json
import requests
import time
from bs4 import BeautifulSoup
from bs4 import Tag

from config.logger import CustomLogger


class WebScraperGCaptain:
    """
    A web scraper for gCaptain news articles.

    This class is responsible for scraping news articles from the gCaptain website.
    It fetches the main page, extracts article and category links, scrapes individual
    article details (such as headline, publication date, author, and content), and stores
    the results in a JSON file.

    Attributes:
        logger (CustomLogger): Logger instance for tracking scraper operations.
        _session (requests.Session): Session object configured with necessary headers for HTTP requests.
        _main_url (str): The base URL of the gCaptain website.
        dataset_dir (str): The directory where scraped data will be stored.
        data_file (str): The file path of the JSON file containing scraped articles.
    """

    def __init__(self):
        """
        Initialize the WebScraperGCaptain instance.

        Sets up logging, creates an HTTP session with custom headers, defines the main URL,
        determines the dataset storage directory, ensures the directory exists, and sets the path
        for the JSON file that will store the scraped articles.
        """
        self.logger = CustomLogger(name="WebScraperGCaptain")
        self.logger.ok("WebScraperGCaptain initialized")
        self._session = self._create_session()
        self._main_url = "https://gcaptain.com/"
        self.dataset_dir = self._get_dataset_directory()
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.data_file = os.path.join(self.dataset_dir, "gcaptain_articles.json")

    def _create_session(self) -> requests.Session:
        """
        Create and configure an HTTP session for requests to the gCaptain website.

        The session is configured with a custom User-Agent, Accept, Accept-Language, and Referer headers.

        Returns:
            requests.Session: A session object with pre-configured headers.
        """
        session = requests.Session()
        session.headers.update({
            'User-Agent': (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/115.0 Safari/537.36"
            ),
            'Accept': "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            'Accept-Language': "en-US,en;q=0.9",
            'Referer': "https://www.google.com/"
        })
        return session

    def _get_dataset_directory(self) -> str:
        """
        Determine the directory where scraped articles will be stored.

        The method checks for the existence of a 'datasets_processed' directory in the current
        or parent directory and constructs the full path for storing gCaptain news articles.

        Returns:
            str: The full directory path for the dataset.
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

        dataset_dir = os.path.join(base_dir, "datasets_processed", "newsarticles", "gcaptain")
        return dataset_dir

    def _standardize_text(self, text: str) -> str:
        """
        Standardize text by normalizing whitespace.

        Replaces multiple whitespace characters with a single space and strips leading
        and trailing whitespace.

        Args:
            text (str): The text to be standardized.

        Returns:
            str: The cleaned text.
        """
        return re.sub(r'\s+', ' ', text).strip()

    def _get_article_links(self, main_url: str) -> list:
        """
        Fetch the main page and extract a list of article and category links.

        The method collects links containing '/category/', '/news/', '/articles/', or '/202'
        from the main page, normalizing relative URLs to absolute ones.

        Args:
            main_url (str): The URL of the main page to fetch.

        Returns:
            list: A list of unique URLs extracted from the main page.
        """
        try:
            response = self._session.get(main_url)
            response.raise_for_status()
        except requests.RequestException as e:
            self.logger.error(f"Error fetching main page of gCaptain: {e}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        links = set()

        # ------------------------------------------------------------------
        # Searching Anchor Tags
        # ------------------------------------------------------------------
        for a in soup.find_all("a", href=True):
            link = a["href"]
            # Normalizing URL
            if link.startswith("/"):
                link = "https://gcaptain.com" + link

            if "/category/" in link:
                links.add(link)
            elif any(x in link for x in ["/news/", "/articles/", "/202"]):
                links.add(link)

        return list(links)

    def _get_articles_from_category_page(self, category_url: str) -> list:
        """
        Extract individual article URLs from a category page.

        Given a category page URL (e.g., /category/news/), this method searches for
        <a> tags with the class "headline" to extract the URLs of individual articles.

        Args:
            category_url (str): The URL of the category page.

        Returns:
            list: A list of article URLs found on the category page.
        """
        try:
            response = self._session.get(category_url)
            response.raise_for_status()
        except requests.RequestException as e:
            self.logger.error(f"Error fetching category page {category_url}: {e}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        links = set()

        # ------------------------------------------------------------------
        # Searching for <a/> Tags containing Headline
        # ------------------------------------------------------------------
        headline_links = soup.find_all("a", class_="headline")
        for link_tag in headline_links:
            href = link_tag.get("href")
            if href:
                if href.startswith("/"):
                    href = "https://gcaptain.com" + href
                links.add(href)
        return list(links)

    def _scrape_article(self, article_url: str) -> dict:
        """
        Scrape an individual article page to extract key details.

        Extracts the headline, publication date, author, and content from the article.
        It looks for:
            - Headline from <h1> tag (often with class "entry-title")
            - Publication date from a <time> tag or meta tags
            - Author from meta tags or elements with "author" in their class name
            - Content from the main article container, after removing unwanted elements

        Args:
            article_url (str): The URL of the article to scrape.

        Returns:
            dict: A dictionary containing article details with keys:
                  'url', 'headline', 'publication_date', 'author', and 'content'.
                  Returns None if an error occurs while fetching the article.
        """
        try:
            response = self._session.get(article_url)
            response.raise_for_status()
        except Exception as e:
            self.logger.error(f"Error fetching article {article_url}: {e}")
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        # ------------------------------------------------------------------
        # Scraping Headline
        # ------------------------------------------------------------------
        headline = None
        h1 = soup.find("h1", class_="entry-title")
        if h1:
            headline = h1.get_text(strip=True)
        else:
            h1 = soup.find("h1")
            if h1:
                headline = h1.get_text(strip=True)

        # ------------------------------------------------------------------
        # Scraping Publication Date
        # ------------------------------------------------------------------
        pub_date = None
        time_tag = soup.find("time")
        if time_tag and time_tag.has_attr("datetime"):
            pub_date = time_tag["datetime"]
        else:
            meta_date = soup.find("meta", property="article:published_time")
            if meta_date and meta_date.get("content"):
                pub_date = meta_date["content"]

        # ------------------------------------------------------------------
        # Scraping Author
        # ------------------------------------------------------------------
        author = None
        meta_author = soup.find("meta", {"name": "author"})
        if meta_author and meta_author.get("content"):
            author = meta_author["content"]
        else:
            author_tag = soup.find(
                lambda tag: tag.name in ["span", "div"]
                            and tag.get("class")
                            and "author" in " ".join(tag.get("class"))
            )
            if author_tag:
                author = author_tag.get_text(strip=True)

        # ------------------------------------------------------------------
        # Extracting Article Content
        # ------------------------------------------------------------------
        content = None

        container = soup.find("div", class_="body")
        if not container:
            container = soup.find("div", class_="td-post-content")
        if not container:
            container = soup.find("div", class_="entry-content")
        if not container:
            container = soup.find("div", class_="article")
        if not container:
            container = soup.find("article")

        if container:

            # ------------------------------------------------------------------
            # Removal of Unwanted Elements
            # ------------------------------------------------------------------
            unwanted_keywords = ["tags", "article-card", "post-nav", "related-articles", "adthrive", "card"]
            for element in container.find_all(True):

                if not element or not hasattr(element, "get"):
                    continue

                try:
                    classes = element.get("class", [])
                except Exception:
                    continue

                if any(any(keyword in cls for keyword in unwanted_keywords) for cls in classes):
                    element.decompose()

            paragraphs = container.find_all("p")
            if paragraphs:
                raw_content = "\n".join(p.get_text(strip=True) for p in paragraphs)
            else:
                raw_content = container.get_text(separator="\n", strip=True)

            # ------------------------------------------------------------------
            # Standardization of Content
            # ------------------------------------------------------------------
            content = self._standardize_text(raw_content)
        else:
            content = ""

        return {
            "url": article_url,
            "headline": headline,
            "publication_date": pub_date,
            "author": author,
            "content": content,
        }

    def _load_existing_articles(self) -> list:
        """
        Load previously scraped articles from the JSON file.

        This helps in avoiding duplicate scraping by retrieving already stored articles.

        Returns:
            list: A list of existing article dictionaries, or an empty list if the file does not exist
                  or an error occurs during loading.
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
        Save the list of articles to a JSON file.

        Args:
            articles (list): A list of article dictionaries to be saved.

        Logs:
            A success message upon saving, or an error message if saving fails.
        """
        try:
            with open(self.data_file, "w", encoding="utf-8") as f:
                json.dump(articles, f, ensure_ascii=False, indent=4)
            self.logger.ok(f"Articles saved to {self.data_file}")
        except Exception as e:
            self.logger.error(f"Error saving articles to JSON: {e}")

    def scrape_gcaptain(self) -> list:
        """
        Scrape the gCaptain website for news articles.

        The scraping process includes:
          1. Fetching the main page and extracting article and category links.
          2. For each category link (e.g., /category/news/), extracting individual article URLs.
          3. Loading previously scraped articles to avoid duplicate work.
          4. Scraping new articles by visiting each new article URL.
          5. Saving the combined list of existing and new articles to a JSON file.

        Returns:
            list: A list containing dictionaries of both existing and newly scraped articles.
        """
        self.logger.info(f"Fetching main page: {self._main_url}")
        initial_links = self._get_article_links(self._main_url)
        self.logger.info(f"Found {len(initial_links)} links on main page")

        article_urls = set()
        for link in initial_links:
            if "/category/" in link:
                self.logger.info(f"Fetching category page: {link}")
                category_article_urls = self._get_articles_from_category_page(link)
                self.logger.info(f"Found {len(category_article_urls)} articles in category listing.")
                article_urls.update(category_article_urls)
                time.sleep(1)
            else:
                article_urls.add(link)

        self.logger.info(f"Total individual articles found: {len(article_urls)}")

        # ------------------------------------------------------------------
        # Loading existing articles to avoid duplicated scraping
        # ------------------------------------------------------------------
        existing_articles = self._load_existing_articles()
        existing_urls = {article.get("url") for article in existing_articles}
        new_article_urls = {url for url in article_urls if url not in existing_urls}
        self.logger.info(f"New articles to scrape: {len(new_article_urls)}")

        articles_data = []
        for idx, article_link in enumerate(new_article_urls, start=1):
            data = self._scrape_article(article_link)
            if data:
                articles_data.append(data)
                self.logger.ok(f"Scraped article: {article_link}")
            time.sleep(1)

        all_articles = existing_articles + articles_data
        self.save_articles_to_json(all_articles)
        return all_articles
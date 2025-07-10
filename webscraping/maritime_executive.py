import os
import json
import requests
import time
from bs4 import BeautifulSoup

from config.logger import CustomLogger


class WebScraperMaritimeExecutive:
    """
    A web scraper for Maritime Executive news articles.

    This scraper fetches the main page of Maritime Executive,
    extracts links that may be direct article URLs or listing pages,
    collects individual article URLs from listing pages, and then
    scrapes each article for its headline, publication date, author,
    and content.

    The scraped articles are merged with previously stored articles
    (serialized as JSON) in the directory:
        Masterthesis-dev/datasets_processed/newsarticles/maritimeexecutive

    When the script is run, it only scrapes new articles (i.e. those whose URLs
    are not already present in the existing dataset).
    """

    def __init__(self):
        """
        Initialize the scraper, its logging, session, and dataset directory.
        """
        self.logger = CustomLogger(name="WebScraperMaritimeExecutive")
        self.logger.ok("WebScraperMaritimeExecutive initialized")
        self._session = self._create_session()
        self._main_url = "https://maritime-executive.com/"

        # Dataset Directory relative to root
        self.dataset_dir = self._get_dataset_directory()
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.data_file = os.path.join(self.dataset_dir, "maritimeexecutive_articles.json")

    def _create_session(self) -> requests.Session:
        """
        Create a requests Session with headers that mimic a real browser.

        Returns:
            requests.Session: A session object with updated headers.
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
        Determine the dataset directory path relative to the project root.

        In a Jupyter Notebook, __file__ is not defined, so this function falls
        back to using os.getcwd().

        Returns:
            str: The absolute path to the dataset directory.
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

        dataset_dir = os.path.join(base_dir, "datasets_processed", "newsarticles", "maritimeexecutive")
        return dataset_dir

    def _get_article_links(self, main_url: str) -> list:
        """
        Fetch the main page and return a list of links.

        These links might be either direct article URLs (containing '/article/')
        or category/listing pages.

        Parameters:
            main_url (str): URL of the main page.

        Returns:
            list: A list of links.
        """
        try:
            response = self._session.get(main_url)
            response.raise_for_status()
        except requests.RequestException:
            self.logger.error("Error fetching main page of Maritime Executive")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.find_all("article")
        links = set()
        for article in articles:
            a_tag = article.find("a", href=True)
            if a_tag:
                link = a_tag["href"]
                if link.startswith("/"):
                    link = "https://maritime-executive.com" + link
                links.add(link)
        return list(links)

    def _get_articles_from_listing_page(self, listing_url: str) -> list:
        """
        Given a URL that points to a listing/category page, extract all the individual
        article URLs. Based on the HTML sample, the listing page contains a div with
        id="article_container" that holds the links.

        Parameters:
            listing_url (str): URL of the listing page.

        Returns:
            list: A list of article URLs extracted from the listing page.
        """
        try:
            response = self._session.get(listing_url)
            response.raise_for_status()
        except requests.RequestException:
            self.logger.error(f"Error fetching listing page {listing_url}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        links = set()
        container = soup.find("div", id="article_container")
        if container:
            for a in container.find_all("a", href=True):
                href = a["href"]
                if "/article/" in href:
                    if href.startswith("/"):
                        href = "https://maritime-executive.com" + href
                    links.add(href)
        else:
            self.logger.info(f"No article container found on listing page: {listing_url}")
        return list(links)

    def _scrape_article(self, article_url: str) -> json:
        """
        Given an article URL, fetch and parse the page for:
          - Headline (from an <h1> tag with class "article-title" if available)
          - Publication date and author (from a <p class="author datePublished"> block)
          - Content (from the article content container)

        Parameters:
            article_url (str): URL of the article to scrape.

        Returns:
            dict: A dictionary containing the scraped data.
        """
        try:
            response = self._session.get(article_url)
            response.raise_for_status()
        except Exception:
            self.logger.error(f"Error fetching article {article_url}")
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        # ------------------------------------------------------------------
        # Scraping Headline
        # ------------------------------------------------------------------
        headline = None
        h1 = soup.find("h1", class_="article-title")
        if h1:
            headline = h1.get_text(strip=True)
        else:
            h1 = soup.find("h1")
            if h1:
                headline = h1.get_text(strip=True)

        # ------------------------------------------------------------------
        # Scraping Publication Date and Authors
        # ------------------------------------------------------------------
        pub_date = None
        author = None
        date_author_tag = soup.find("p", class_="author datePublished")
        if date_author_tag:
            text_content = date_author_tag.get_text(" ", strip=True)
            parts = text_content.split("by")
            if len(parts) >= 2:
                pub_date = parts[0].replace("Published", "").strip()
                a_tag = date_author_tag.find("a")
                if a_tag:
                    author = a_tag.get_text(strip=True)
        else:
            time_tag = soup.find("time")
            if time_tag and time_tag.has_attr("datetime"):
                pub_date = time_tag["datetime"]
            meta_author = soup.find("meta", attrs={"name": "author"})
            if meta_author and meta_author.has_attr("content"):
                author = meta_author["content"]

        # ------------------------------------------------------------------
        # Scrape Article Content
        # ------------------------------------------------------------------
        content = None
        container = soup.find("div", id="article-container")
        if container:
            paragraphs = container.find_all("p")
            if paragraphs:
                content = "\n".join(p.get_text(strip=True) for p in paragraphs)
            else:
                content = container.get_text(separator="\n", strip=True)
        else:
            possible_selectors = [
                {"name": "div", "class": "article-content"},
                {"name": "div", "class": "entry-content"},
                {"name": "div", "class": "post-content"},
                {"name": "div", "class": "article__content"},
                {"name": "div", "class": "article-body"},
                {"name": "div", "class": "content"},
                {"name": "article"},
            ]
            for sel in possible_selectors:
                if "class" in sel:
                    container = soup.find(sel["name"], class_=sel["class"])
                else:
                    container = soup.find(sel["name"])
                if container:
                    paragraphs = container.find_all("p")
                    if paragraphs:
                        content = "\n".join(p.get_text(strip=True) for p in paragraphs)
                    else:
                        content = container.get_text(separator="\n", strip=True)
                    if content and len(content) > 50:
                        break

        return {
            "url": article_url,
            "headline": headline,
            "publication_date": pub_date,
            "author": author,
            "content": content,
        }

    def _load_existing_articles(self) -> list:
        """
        Load existing scraped articles from the JSON file.

        Returns:
            list: A list of existing article dictionaries.
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
        Save the given list of articles to a JSON file in the dataset directory.

        Parameters:
            articles (list): A list of article dictionaries to save.
        """
        try:
            with open(self.data_file, "w", encoding="utf-8") as f:
                json.dump(articles, f, ensure_ascii=False, indent=4)
            self.logger.ok(f"Articles saved to {self.data_file}")
        except Exception as e:
            self.logger.error(f"Error saving articles to JSON: {e}")

    def scrape_maritime_executive(self) -> list:
        """
        Scrape the Maritime Executive website for news articles.

        The method fetches the main page, extracts links (either direct articles or listing pages),
        collects individual article URLs from listing pages, and then scrapes each article.
        It then loads any existing articles from the dataset and only scrapes new articles.

        Returns:
            list: A list of dictionaries containing the merged scraped news articles.
        """
        self.logger.info(f"Fetching main page: {self._main_url}")
        initial_links = self._get_article_links(self._main_url)
        self.logger.info(f"Found {len(initial_links)} links on main page")

        # ------------------------------------------------------------------
        # Collection of individual articles
        # ------------------------------------------------------------------
        article_urls = set()
        for link in initial_links:
            if "/article/" in link:
                article_urls.add(link)
            else:
                self.logger.info(f"Fetching listing page: {link}")
                listing_article_urls = self._get_articles_from_listing_page(link)
                self.logger.info(f"Found {len(listing_article_urls)} articles in listing.")
                article_urls.update(listing_article_urls)
                time.sleep(1)

        self.logger.info(f"Total individual articles found: {len(article_urls)}")

        # ------------------------------------------------------------------
        # Loading of existing URLS
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
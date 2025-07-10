import requests
from bs4 import BeautifulSoup
import time
import os
import re
import json
from typing import Any, Dict, List, Set, Tuple
from urllib.parse import urljoin, unquote
from collections import Counter
from config.logger import CustomLogger


class WikipediaCharteringScraper:

    def __init__(self, output_dir=None):

        self.logger = CustomLogger(name="WikipediaCharteringScraper")
        self.base_url = "https://en.wikipedia.org"
        self.visited_urls = set()
        self.url_cache = {}

        # -------------------------------------------------------------------
        # Output Directory Setup
        # -------------------------------------------------------------------
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = os.path.join(os.getcwd(), 'corpus')
        os.makedirs(self.output_dir, exist_ok=True)

        # -------------------------------------------------------------------
        # Blocked Categories and Patterns - Prevention of Topic Drift
        # -------------------------------------------------------------------
        self.blocked_categories = {
            'Category:Racehorses',
            'Category:Thoroughbred',
            'Category:Horse',
            'Category:Horses',
            'Category:Individual_horses',
            'Category:Fictional_horses',
            'Category:Horse_racing',
            'Category:People',
            'Category:Biography',
            'Category:Films',
            'Category:Television',
            'Category:Music',
            'Category:Books',
            'Category:Novels',
            'Category:Video_games',
            'Category:Artists',
            'Category:Politicians',
            'Category:Sportspeople',
            'Category:Actors',
            'Category:Medicine',
            'Category:Medical_terms',
            'Category:Health',
            'Category:Diseases',
            'Category:Anatomy',
            'Category:Physiology',
            'Category:Surgery',
            'Category:Healthcare',
            'Category:Physicians',
            'Category:Hospitals',
            'Category:Biology',
            'Category:Physics',
            'Category:Chemistry',
            'Category:Astronomy',
            'Category:Religion',
            'Category:Philosophy',
            'Category:Military_history',
            'Category:Military_operations',
            'Category:Fiction',
            'Category:Literature',
            'Category:Mythology',
            'Category:Gaming',
            'Category:Sports',
        }

        self.blocked_url_patterns = [
            r'_\(horse\)',
            r'_\(racehorse\)',
            r'_\(footballer\)',
            r'_\(actor\)',
            r'_\(musician\)',
            r'_\(album\)',
            r'_\(film\)',
            r'_\(TV_series\)',
            r'_\(singer\)',
            r'_\(writer\)',
            r'_\(disease\)',
            r'_\(medicine\)',
            r'_\(medical\)',
            r'_\(biology\)',
            r'_\(anatomy\)',
            r'_\(military\)',
            r'_\(novel\)',
            r'_\(mythology\)',
            r'_\(religion\)',
        ]

        # -------------------------------------------------------------------
        # Primary Maritime Domain Terms
        # -------------------------------------------------------------------
        self.primary_maritime_terms = {
            # Core shipping terms
            "charterparty", "ship chartering", "voyage charter", "time charter",
            "bareboat charter", "demurrage", "laytime", "bill of lading", "shipping",
            "maritime transport", "vessel", "tanker ship", "bulk carrier", "cargo ship",
            "container ship", "freight transport", "maritime law", "shipping market",
            "bunkering", "deadweight tonnage", "gross tonnage",

            # Ports and Infrastructure
            "seaport", "container terminal", "shipping container", "port authority",
            "berth", "quay", "port management", "dock", "jetty", "pier", "wharf",

            # Shipping Operations
            "stevedore", "ship management", "shipbroker", "freight rate", "seafarer",
            "maritime safety", "ship operation", "marine insurance", "protection and indemnity",
            "bunker fuel", "cargo handling", "stowage", "containerization",

            # Maritime Legal Terms
            "admiralty law", "maritime lien", "general average", "salvage",
            "carriage of goods", "hague rules", "hague-visby", "rotterdam rules",
            "incoterms", "charterer", "shipowner", "carriage of goods by sea",

            # Maritime Organizations
            "imo", "international maritime organization", "classification society",
            "flag state", "bimco", "lloyd's register", "baltic exchange",

            # Major Shipping Routes
            "suez canal", "panama canal", "strait of malacca", "strait of hormuz",
        }

        # -------------------------------------------------------------------
        # Secondary Maritime Terms
        # -------------------------------------------------------------------
        self.secondary_maritime_terms = {
            # Vessel types
            "ship", "boat", "vessel", "tanker", "bulk", "carrier", "container", "cargo",
            "reefer", "ro-ro", "cruise", "ferry", "barge", "tug", "offshore",
            "lng carrier", "vlcc", "panamax", "aframax", "suezmax", "capesize", "handymax",

            # Shipping activities and concepts
            "shipping", "maritime", "freight", "port", "harbor", "dock", "terminal",
            "tonnage", "naval", "marine", "nautical", "navigation", "voyage", "charter",
            "logistics", "shipment", "transport", "cargo", "stevedore", "consignee",
            "bill of lading", "charterparty", "laytime", "demurrage", "broker", "bunker",
            "freight", "logistics", "loading", "unloading", "storage", "warehousing",

            # Maritime geography
            "sea", "ocean", "strait", "channel", "gulf", "bay", "seaway", "waterway",
            "lane", "route", "passage", "canal", "lock", "berth", "anchorage", "roadstead",

            # Maritime infrastructure
            "port", "harbor", "dock", "terminal", "berth", "pier", "wharf", "quay",
            "jetty", "shipyard", "dry dock", "slipway", "crane", "warehouse", "terminal",

            # Maritime business terms
            "shipowner", "charterer", "broker", "agent", "consignee", "shipper", "carrier",
            "freight", "insurance", "classification", "registry", "flag", "registration",
        }

        # -------------------------------------------------------------------
        # Maritime Industry Entities
        # -------------------------------------------------------------------
        self.maritime_entities = {
            # Major shipping companies
            "maersk", "msc", "cma cgm", "cosco", "hapag-lloyd", "one", "evergreen",
            "yang ming", "hmm", "pil", "zim", "wan hai", "safmarine", "hamburg süd",

            # Major ports
            "port of shanghai", "port of singapore", "port of rotterdam", "port of antwerp",
            "port of hong kong", "port of busan", "port of hamburg", "port of los angeles",
            "port of long beach", "port of new york", "port of new jersey", "jebel ali port",

            # Regulatory bodies
            "international maritime organization", "lloyd's register", "dnv gl", "bureau veritas",
            "american bureau of shipping", "china classification society", "nippon kaiji kyokai",
            "bimco", "intertanko", "intercargo", "international chamber of shipping",
            "unctad", "paris mou", "tokyo mou", "maritime and coastguard agency",

            # Industry associations
            "baltic exchange", "fonasba", "london maritime arbitrators association",
            "federation of national associations of ship brokers and agents",
        }

        self.commodity_primary_terms = {
            # Commodity trading core terms
            "commodity trading", "futures contract", "forward contract", "commodity market",
            "dry bulk", "wet bulk", "charter market", "spot market", "freight derivatives",
            "commodity exchange", "trading desk", "forward freight agreement", "ffa",
            "baltic dry index", "baltic dirty tanker index", "commodity price", "freight index",
            "contango", "backwardation", "commodity futures", "commodity pricing",

            # Contract and chartering terms
            "asbatankvoy", "shellvoy", "bimchemvoy", "gencon", "nype 93", "baltime",
            "boxtime", "supplytime", "voyage charter", "fixture", "laycan", "free in out",
            "notice of readiness", "worldscale", "baltic exchange", "vessel fixture",

            # Important benchmark indices
            "platts", "argus media", "ice brent", "wti crude", "baltic indices",
        }
        self.primary_maritime_terms.update(self.commodity_primary_terms)

        self.commodity_secondary_terms = {
            # Specific commodities shipped in bulk
            "crude oil", "iron ore", "coal", "bauxite", "grain", "wheat", "corn",
            "soybean", "rice", "sugar", "palm oil", "vegetable oil", "fertilizer",
            "phosphate", "sulphur", "steel", "scrap metal", "copper", "aluminium",
            "zinc", "nickel", "lng", "lpg", "naphtha", "jet fuel", "gasoline", "diesel",
            "ammonia", "urea", "petcoke", "cement", "timber", "wood pulp",

            # Commodity trading vocabulary
            "price", "benchmark", "physical delivery", "settlement", "trading", "trader",
            "contract", "futures", "hedging", "arbitrage", "derivative", "broker",
            "clearing", "exchange", "otc", "option", "swap", "position", "cargo",
            "consignment", "load port", "discharge port", "laydays", "cancelling",
            "storage", "terminal", "tank farm", "warehouse", "silo", "berth",
        }
        self.secondary_maritime_terms.update(self.commodity_secondary_terms)

        self.commodity_entities = {
            # Commodity trading companies
            "glencore", "vitol", "trafigura", "cargill", "adm", "bunge", "louis dreyfus",
            "mercuria", "gunvor", "noble group", "cofco", "wilmar", "koch industries",

            # Commodity exchanges
            "london metal exchange", "lme", "ice futures", "cme group", "nymex",
            "chicago board of trade", "cbot", "dalian commodity exchange",
            "shanghai futures exchange", "tokyo commodity exchange", "tocom",
            "singapore exchange", "sgx", "euronext", "matif",

            # Price reporting agencies
            "platts", "argus media", "opis", "fastmarkets", "icis",
        }
        self.maritime_entities.update(self.commodity_entities)

        # -------------------------------------------------------------------
        # Lookup Dataset
        # -------------------------------------------------------------------
        self.all_maritime_terms = set()
        self.all_maritime_terms.update(self.primary_maritime_terms)
        self.all_maritime_terms.update(self.secondary_maritime_terms)
        self.all_maritime_terms.update(self.maritime_entities)

        self.all_maritime_terms = {term.lower() for term in self.all_maritime_terms}
        self.primary_maritime_terms = {term.lower() for term in self.primary_maritime_terms}
        self.maritime_entities = {entity.lower() for entity in self.maritime_entities}

        # -------------------------------------------------------------------
        # Visited URLS File
        # -------------------------------------------------------------------
        self.urls_log_file = os.path.join(self.output_dir, "visited_urls.json")
        if os.path.exists(self.urls_log_file):
            with open(self.urls_log_file, 'r') as f:
                self.visited_urls = set(json.load(f))

        # -------------------------------------------------------------------
        # Loading Existing Articles
        # -------------------------------------------------------------------
        self.json_output_file = os.path.join(self.output_dir, "maritime_articles.json")
        self.articles = []

        if os.path.exists(self.json_output_file):
            try:
                with open(self.json_output_file, 'r', encoding='utf-8') as f:
                    self.articles = json.load(f)
                    self.logger.info(f"Loaded {len(self.articles)} existing articles from JSON file")
            except json.JSONDecodeError:
                self.logger.warning(f"Could not load existing articles from JSON file. Starting fresh.")

    def is_blocked_category(self, url: str) -> bool:
        """Check if URL is for a blocked category"""
        for blocked_cat in self.blocked_categories:
            if blocked_cat.lower().replace('_', ' ') in url.lower():
                return True
        return False

    def matches_blocked_pattern(self, url: str) -> bool:
        """Check if URL matches a blocked pattern"""
        for pattern in self.blocked_url_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        return False

    def normalize_url(self, url: str) -> str:
        """Normalize URL to avoid duplicates with minor differences"""
        url = url.lower()

        url = url.rstrip('/')

        url = url.split('#')[0]

        url = unquote(url)

        if '/wiki/' in url:
            base, title = url.split('/wiki/', 1)
            title = title.replace('_', ' ')
            title = re.sub(r'\s*\([^)]+\)\s*$', '', title)
            url = f"{base}/wiki/{title}"

        return url

    def is_url_visited(self, url: str) -> bool:
        """Check if URL has been visited, using normalized form"""
        normalized = self.normalize_url(url)
        return normalized in self.url_cache

    def calculate_maritime_relevance(self, text: str) -> Tuple[float, int, Set[str]]:
        """
        Calculate how relevant a text is to maritime/shipping domain
        Returns:
            - relevance score (0.0 to 1.0)
            - count of primary maritime terms
            - set of found maritime terms
        """
        if not text:
            return 0.0, 0, set()

        text = text.lower()

        medical_terms = ["patient", "disease", "treatment", "hospital", "doctor",
                         "surgery", "blood vessel", "heart", "medicine", "physician",
                         "diagnosis", "clinical", "therapy", "medical"]

        medical_context = sum(1 for term in medical_terms if term in text)
        if medical_context >= 2:
            return 0.01, 0, set()

        found_terms = set()
        primary_term_count = 0
        entity_count = 0

        for term in self.primary_maritime_terms:
            if term in text:
                found_terms.add(term)
                primary_term_count += text.count(term)

        for entity in self.maritime_entities:
            if entity in text:
                found_terms.add(entity)
                entity_count += text.count(entity)

        secondary_term_count = 0
        for term in self.secondary_maritime_terms:
            if term in text:
                found_terms.add(term)
                secondary_term_count += text.count(term)

        total_words = len(text.split())
        if total_words == 0:
            return 0.0, 0, found_terms

        # Scoring Relatedness to Chartering
        score = (primary_term_count * 3 + entity_count * 2 + secondary_term_count) / (total_words + 100)
        score = min(score * 20, 1.0)

        return score, primary_term_count, found_terms

    def is_maritime_related(self, text: str, min_score: float = 0.035, min_primary_terms: int = 1) -> bool:
        """
        Determine if text is related to maritime/shipping domain with better accuracy
        Uses a combined approach of term density and primary term presence
        """
        relevance_score, primary_term_count, found_terms = self.calculate_maritime_relevance(text)
        return relevance_score >= min_score and primary_term_count >= min_primary_terms

    def get_page_content(self, url: str) -> str:
        """Fetch and parse a Wikipedia page"""
        headers = {
            'User-Agent': 'MaritimeResearchBot/1.0 (research project on maritime terminology; contact@example.com)'
        }

        try:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                self.logger.warning(f"Failed to fetch {url}: HTTP {response.status_code}")
                return None

            return response.text
        except Exception as e:
            self.logger.error(f"Error fetching {url}: {e}")
            return None

    def parse_article(self, url: str) -> tuple[Dict[str, str | Any], Any]:
        """Parse Wikipedia article and extract useful text with improved topic filtering"""
        if self.matches_blocked_pattern(url):
            self.logger.warning(f"Skipping URL with blocked pattern: {url}")
            return None, []

        content = self.get_page_content(url)
        if not content:
            return None, []

        soup = BeautifulSoup(content, 'html.parser')

        # -------------------------------------------------------------------
        # Article Title
        # -------------------------------------------------------------------
        title = soup.find(id="firstHeading")
        if not title:
            title_text = "Unknown"
            maritime_title = False
        else:
            title_text = title.text.strip()
            maritime_keywords = ["port of ",
                                 "shipping",
                                 "maritime",
                                 "vessel",
                                 "cargo",
                                 "tanker",
                                 "container",
                                 "seaport",
                                 "harbor",
                                 "harbour",
                                 "terminal",
                                 "dock",
                                 "wharf",
                                 "bulk carrier",
                                 "shipyard",
                                 "freight",
                                 "charter",
                                 "ocean",
                                 "tonnage",
                                 "shipowner"]

            maritime_title = any(keyword in title_text.lower() for keyword in maritime_keywords)
            if maritime_title:
                self.logger.warning(f"Auto-accepting maritime title: {title_text}")

        # -------------------------------------------------------------------
        # Category Check
        # -------------------------------------------------------------------
        categories = []
        category_links = soup.select('div#mw-normal-catlinks ul li a')
        for cat_link in category_links:
            categories.append(cat_link.text.strip())

        for category in categories:
            for blocked in self.blocked_categories:
                blocked_name = blocked.replace('Category:', '').replace('_', ' ').lower()
                if blocked_name in category.lower():
                    self.logger.warning(f"Skipping article in blocked category: {title_text} - {category}")
                    return None, []

        # -------------------------------------------------------------------
        # Check for Maritime Categories
        # -------------------------------------------------------------------
        maritime_categories = ["shipping", "port", "maritime", "transport", "logistics",
                               "navigation", "nautical", "naval architecture", "merchant",
                               "waterborne", "vessel", "boat", "terminal", "dock", "ship"]

        category_is_maritime = any(
            any(keyword in category.lower() for keyword in maritime_categories)
            for category in categories
        )

        # -------------------------------------------------------------------
        # Assessing Relevance for Shipping
        # -------------------------------------------------------------------
        main_content = soup.find(id="bodyContent")
        if not main_content:
            return None, []

        if not (maritime_title or category_is_maritime):
            lead_paragraphs = main_content.select('.mw-parser-output > p')
            lead_text = ""
            for p in lead_paragraphs[:3]:
                lead_text += p.get_text() + "\n\n"

            if "port" in url.lower() or "terminal" in url.lower() or "harbour" in url.lower() or "harbor" in url.lower():
                port_threshold = 0.015
                port_min_terms = 1

                if not self.is_maritime_related(lead_text, min_score=port_threshold, min_primary_terms=port_min_terms):
                    more_text = ""
                    for p in main_content.select('.mw-parser-output > p')[:10]:
                        more_text += p.get_text() + "\n\n"

                    if not self.is_maritime_related(more_text, min_score=port_threshold,
                                                    min_primary_terms=port_min_terms):
                        self.logger.warning(f"Skipping non-maritime port article: {title_text}")
                        return None, []
            else:
                if not self.is_maritime_related(lead_text):
                    more_text = ""
                    for p in main_content.select('.mw-parser-output > p')[:10]:
                        more_text += p.get_text() + "\n\n"

                    if not self.is_maritime_related(more_text, min_score=0.015, min_primary_terms=1):
                        self.logger.warning(f"Skipping non-maritime article: {title_text}")
                        return None, []

        # -------------------------------------------------------------------
        # Main Text Extraction
        # -------------------------------------------------------------------
        article_text = ""
        main_content = soup.find(id="mw-content-text")

        if main_content:
            for element in main_content.select(
                    'table, .mw-editsection, .reference, .reflist, .navbox, .noprint, script, style'):
                element.extract()

            paragraphs = main_content.select('.mw-parser-output > p')
            for p in paragraphs:
                article_text += p.get_text() + "\n\n"

            headers = main_content.select('.mw-parser-output > h2, .mw-parser-output > h3, .mw-parser-output > h4')
            for header in headers:
                header_text = header.get_text().strip()
                if header_text and 'References' not in header_text and 'See also' not in header_text and 'External links' not in header_text:
                    article_text += f"\n\n{header_text}\n\n"

                    current = header.next_sibling
                    while current and not current.name in ['h2', 'h3', 'h4']:
                        if current.name == 'p':
                            article_text += current.get_text() + "\n\n"
                        current = current.next_sibling

        # -------------------------------------------------------------------
        # Maritime Relevance Calculation
        # -------------------------------------------------------------------
        full_relevance, term_count, found_terms = self.calculate_maritime_relevance(article_text)

        if maritime_title:
            full_relevance = max(full_relevance, 0.7)

        if category_is_maritime:
            full_relevance = max(full_relevance, 0.5)

        if "port of" in title_text.lower() or any(
                p in title_text.lower() for p in ["harbor", "harbour", "terminal", "dock"]):
            full_relevance = max(full_relevance, 0.8)

        # -------------------------------------------------------------------
        # Snowball Search of Related Links
        # -------------------------------------------------------------------
        related_links = []

        if main_content:
            content_links = main_content.select('a[href^="/wiki/"]')

            max_links = 50 if full_relevance > 0.1 else 20
            link_counter = 0

            for link in content_links:
                href = link.get('href')
                if any(x in href for x in [':',
                                           '#',
                                           'Main_Page',
                                           'Wikipedia:',
                                           'Special:',
                                           'Talk:',
                                           'Help:']):
                    continue

                if self.matches_blocked_pattern(href):
                    continue

                link_text = link.get_text().strip()
                if not link_text:
                    continue

                if full_relevance > 0.25:
                    is_relevant = any(term in link_text.lower() for term in self.all_maritime_terms)
                else:
                    is_relevant = any(term in link_text.lower() for term in self.primary_maritime_terms)

                if is_relevant:
                    full_url = urljoin(self.base_url, href)
                    related_links.append(full_url)
                    link_counter += 1
                    if link_counter >= max_links:
                        break

        see_also = soup.find(id="See_also")
        if see_also and see_also.parent:
            see_also_section = see_also.parent.find_next('ul')
            if see_also_section:
                for link in see_also_section.select('a[href^="/wiki/"]'):
                    href = link.get('href')
                    if any(x in href for x in [':', '#', 'Main_Page']):
                        continue
                    if self.matches_blocked_pattern(href):
                        continue
                    full_url = urljoin(self.base_url, href)
                    related_links.append(full_url)

        # -------------------------------------------------------------------
        # Category Links
        # -------------------------------------------------------------------
        category_link_elements = soup.select('div#mw-normal-catlinks ul li a')
        for link in category_link_elements:
            href = link.get('href')
            cat_text = link.get_text().strip()

            if 'Category:' in href and (
                    any(mar_term in cat_text.lower() for mar_term in self.all_maritime_terms) or
                    "transport" in cat_text.lower() or
                    "shipping" in cat_text.lower() or
                    "maritime" in cat_text.lower() or
                    "naval" in cat_text.lower() or
                    "port" in cat_text.lower()
            ) and not self.is_blocked_category(href):
                category_url = urljoin(self.base_url, href)
                related_links.append(category_url)

        article_data = {
            'title': title_text,
            'url': url,
            'content': article_text,
            'categories': categories,
            'maritime_terms': list(found_terms),
            'maritime_relevance': full_relevance,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        return article_data, related_links

    def parse_category_page(self, url: str) -> List[str]:
        """Parse a category page and extract article links with improved filtering"""
        content = self.get_page_content(url)
        if not content:
            return []

        if self.is_blocked_category(url):
            self.logger.warning(f"Skipping blocked category: {url}")
            return []

        soup = BeautifulSoup(content, 'html.parser')
        related_links = []

        category_header = soup.find(id="firstHeading")
        if category_header:
            category_name = category_header.text.strip().replace('Category:', '')

            for blocked in self.blocked_categories:
                blocked_name = blocked.replace('Category:', '').replace('_', ' ').lower()
                if blocked_name in category_name.lower():
                    self.logger.warning(f"Skipping blocked category: {category_name}")
                    return []

            is_maritime_category = any(term in category_name.lower() for term in self.all_maritime_terms)
            if not is_maritime_category and not any(x in category_name.lower() for x in
                                                    ['ship', 'boat', 'maritime', 'port', 'shipping', 'transport',
                                                     'cargo', 'naval', 'ocean', 'sea']):
                self.logger.warning(f"Skipping non-maritime category: {category_name}")
                return []

        members = soup.select('div#mw-pages a[href^="/wiki/"]')
        for link in members:
            href = link.get('href')
            if ':' not in href and not self.matches_blocked_pattern(href):
                full_url = urljoin(self.base_url, href)
                related_links.append(full_url)

        subcats = soup.select('div#mw-subcategories a[href^="/wiki/Category:"]')
        for link in subcats:
            href = link.get('href')
            link_text = link.get_text().strip()

            if self.is_blocked_category(href):
                continue

            if any(term in link_text.lower() for term in self.all_maritime_terms) or \
                    any(x in link_text.lower() for x in ['ship', 'boat', 'maritime', 'port',
                                                         'shipping', 'transport', 'cargo', 'naval',
                                                         'ocean', 'sea', 'vessel', 'marine']):
                full_url = urljoin(self.base_url, link.get('href'))
                related_links.append(full_url)

        return related_links

    def save_articles_to_json(self) -> None:
        """Save all collected articles to a JSON file"""
        try:
            with open(self.json_output_file, 'w', encoding='utf-8') as f:
                json.dump(self.articles, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved {len(self.articles)} articles to JSON file")
        except Exception as e:
            self.logger.error(f"Error saving articles to JSON: {e}")

    def collect_article(self, article_data: dict) -> None:
        """Add article to the collection instead of saving as individual file"""
        if not article_data or not article_data.get('content'):
            return

        title = article_data['title']

        for existing in self.articles:
            if existing.get('url') == article_data.get('url'):
                self.logger.warning(f"Skipping duplicate article: {title}")
                return

        self.articles.append(article_data)
        self.logger.ok(f"Collected article: {title} (Relevance: {article_data.get('maritime_relevance', 0):.2f})")

        if len(self.articles) % 10 == 0:
            self.save_articles_to_json()

    def save_visited_urls(self) -> None:
        """Save the set of visited URLs to a file"""
        with open(self.urls_log_file, 'w') as f:
            json.dump(list(self.visited_urls), f)

    def get_domain_statistics(self) -> Dict:
        """Get statistics about the collected articles and maritime terms"""
        if not self.articles:
            return {"status": "No articles collected"}

        total_articles = len(self.articles)
        total_words = sum(len(article.get('content', '').split()) for article in self.articles)

        term_counts = Counter()
        for article in self.articles:
            terms = article.get('maritime_terms', [])
            for term in terms:
                term_counts[term] += 1

        relevance_scores = [article.get('maritime_relevance', 0) for article in self.articles]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0

        top_terms = term_counts.most_common(20)

        return {
            "total_articles": total_articles,
            "total_words": total_words,
            "average_relevance": avg_relevance,
            "top_maritime_terms": top_terms
        }

    def crawl(self, seed_urls: list, max_pages: int = 300) -> None:
        """Crawl Wikipedia starting from seed URLs with improved topic control"""

        to_visit = []

        # Appending Seed URLS
        for url in seed_urls.copy():
            if url not in to_visit:
                to_visit.append(url)

        page_count = 0

        start_time = time.time()

        processed_count = 0
        collected_count = 0
        category_count = 0

        while to_visit and page_count < max_pages:
            if processed_count % 20 == 0 and processed_count > 0:
                elapsed = time.time() - start_time
                pages_per_hour = (processed_count * 3600) / elapsed if elapsed > 0 else 0
                self.logger.info(f"Progress: Processed {processed_count} pages, collected {collected_count} articles")
                self.logger.info(f"Average speed: {pages_per_hour:.1f} pages/hour")

                if pages_per_hour > 0:
                    remaining = max_pages - page_count
                    estimated_hours = remaining / pages_per_hour
                    self.logger.info(f"Estimated time to completion: {estimated_hours:.1f} hours")

            current_url = to_visit.pop(0)

            # -------------------------------------------------------------------
            # Skip Visited/Blocked URLs
            # -------------------------------------------------------------------
            if self.matches_blocked_pattern(current_url):
                continue

            # -------------------------------------------------------------------
            # Skip Visited
            # -------------------------------------------------------------------
            if self.is_url_visited(current_url):
                continue

            self.logger.info(f"Processing {processed_count + 1}/{max_pages}: {current_url}")
            self.visited_urls.add(current_url)
            processed_count += 1

            # -------------------------------------------------------------------
            # Handling of Category Pages
            # -------------------------------------------------------------------
            if 'Category:' in current_url:
                if self.is_blocked_category(current_url):
                    continue

                self.logger.info(f"Parsing category: {current_url}")
                new_links = self.parse_category_page(current_url)
                category_count += 1

                max_category_links = 15
                for link in new_links[:max_category_links]:
                    if link not in self.visited_urls and link not in to_visit:
                        to_visit.append(link)

                self.save_visited_urls()
                time.sleep(1)
                continue

            article_data, related_links = self.parse_article(current_url)

            if article_data and article_data.get('content'):
                content_length = len(article_data['content'].split())

                if content_length < 20:
                    self.logger.warning(f"Skipping short article: {article_data['title']} ({content_length} words)")

                    relevance = article_data.get('maritime_relevance', 0)
                    max_links = min(int(10 + relevance * 20), 20)

                    for link in related_links[:max_links]:
                        if link not in self.visited_urls and link not in to_visit:
                            to_visit.append(link)

                    continue

                self.collect_article(article_data)
                page_count += 1
                collected_count += 1

                relevance = article_data.get('maritime_relevance', 0)

                max_links = min(int(10 + relevance * 20), 20)

                for link in related_links[:max_links]:
                    if link not in self.visited_urls and link not in to_visit:
                        to_visit.append(link)

            # -------------------------------------------------------------------
            # Periodic Progress Saving
            # -------------------------------------------------------------------
            if processed_count % 10 == 0:
                self.save_visited_urls()

            time.sleep(1)

        self.save_visited_urls()
        self.save_articles_to_json()

        elapsed_time = (time.time() - start_time) / 60
        self.logger.ok(f"Crawling complete. Scraped {page_count} pages in {elapsed_time:.1f} minutes.")
        self.logger.ok(f"Processed {processed_count} pages total, including {category_count} categories.")

        stats = self.get_domain_statistics()
        self.logger.info(f"Corpus Statistics: {stats}")

    def create_training_corpus(self, output_file='chartering_corpus.json'):
        """Create a simplified corpus suitable for model training"""
        training_data = []

        sorted_articles = sorted(self.articles, key=lambda x: x.get('maritime_relevance', 0), reverse=True)

        for article in sorted_articles:
            if article.get('maritime_relevance', 0) < 0.05:
                continue

            training_data.append({
                'text': article['content'],
                'title': article['title'],
                'url': article['url'],
                'relevance': article.get('maritime_relevance', 0)
            })

        corpus_path = os.path.join(self.output_dir, output_file)
        with open(corpus_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)

        self.logger.ok(f"Created training corpus with {len(training_data)} articles at {corpus_path}")
        return corpus_path
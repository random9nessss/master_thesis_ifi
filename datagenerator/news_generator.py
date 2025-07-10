from quotegenerator.quote_generator import FreightQuoteEngine
from attributes.news_attribute_sampler import CharteringNewsAttributeSampler
from config.logger import CustomLogger


class CharteringNewsGenerator:
    """
    Constructs a prompt to instruct an LLM to generate maritime chartering news items.
    Uses the news attributes from CharteringNewsAttributeSampler and an optional
    FreightQuoteEngine to supply numeric or anecdotal freight data.
    """

    def __init__(self, sampler: CharteringNewsAttributeSampler):
        self._sampler = sampler

        self.logger = CustomLogger(name="CharteringNewsGenerator")
        self.logger.ok("CharteringNewsGenerator initialized")

    def construct_prompt(self) -> str:
        """
        Builds a prompt string that instructs an LLM to generate chartering news items
        in a structured JSON format. The prompt can be sent to the LLM of your choice.
        """

        # ------------------------------------------------------------------
        # Attribute Sampling
        # ------------------------------------------------------------------
        attributes = self._sampler.sample_random_attributes()

        category = attributes["category"]
        region = attributes["region"]
        impact = attributes["impact"]
        tone = attributes["tone"]
        source = attributes["source"]
        journalist = attributes["journalist"]
        article_count = attributes["article_count"]

        # ------------------------------------------------------------------
        # Guidance Prompt
        # ------------------------------------------------------------------
        prompt = f"""
            You are a creative language model tasked with generating **maritime chartering news** as a JSON array. 
            Each element in the JSON array should represent a **single news article** with the following attributes:
            1. **"headline"**:      A concise yet informative title for the article.
            2. **"timestamp"**:     A realistic date/time stamp in chronological order (e.g., "2025-05-10 08:30").
            3. **"source"**:        The name of the publishing newspaper,"{source}".
            4. **"region"**:        The main geographical focus, "{region}".
            5. **"impact"**:        The overall assessment of the news impact, "{impact}".
            6. **"tone"**:          A descriptor for how the news is presented, "{tone}".
            7. **"category"**:      The overarching category, "{category}".
            8. **"journalist"**:    The author of the article, "{journalist}".
            9. **"body"**:          A short summary or highlight of the chartering news story (one to three paragraphs).

            **Overall Theme & Guidance**:
            - Category:             **{category}**
            - Primary Region:       **{region}**
            - Impact Assessment:    **{impact}**
            - Tone of Coverage:     **{tone}** 
            - Source Name:          **{source}**
            - Journalist:           **{journalist}**

            - Each article **may reference** shipping data, trends, or anecdotal evidence 

            - The output must be a **pure JSON array** containing about {article_count} articles. 

            **Important**:
            - Keep the article bodies concise, but ensure they offer enough substance to feel 
              like a real news item. 
            - The **timestamp** in each article should reflect a chronological order 
              (e.g., increment by minutes, hours, or days).

            Please produce the final output **exclusively** in JSON, with no markdown formatting or explanations.
            
        """.strip()

        self.logger.info("Constructed chartering news prompt")
        return prompt
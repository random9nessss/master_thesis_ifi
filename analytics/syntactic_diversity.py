import spacy
import pandas as pd
from typing import List, Tuple
import json

spacy.cli.download("en_core_web_sm")


class SyntacticDiversity:
    """
    A class to compute the syntactic diversity of email texts based on parse-tree distances.

    Syntactic diversity is defined as the average pairwise tree edit distance between the parse trees
    (represented as lists of dependency edges) of the texts. A higher average distance indicates greater
    syntactic variability.

    This class can be instantiated either by providing a list of texts directly (via the `texts` parameter)
    or by providing a file path to a JSON file containing email data (via the `file_path` parameter) along with
    a field indicator (either "body" or "subject") to extract the desired text from each email chain.

    Note:
        The tree edit distance is implemented as a placeholder function that simply computes the absolute
        difference in the number of edges between two parse trees. For a robust measure, consider replacing
        this with an implementation of the Zhang-Shasha algorithm or another tree edit distance algorithm.
    """

    def __init__(self, file_path: str = None, texts: List[str] = None, field: str = "body"):
        """
        Initialize the SyntacticDiversity object.

        Args:
            file_path (str, optional): Path to the JSON file containing email data.
                                        If provided, texts will be extracted from the file using the specified field.
            texts (List[str], optional): A list of text strings to analyze. Used if file_path is not provided.
            field (str, optional): The field to analyze from each email when file_path is provided.
                                   Must be either "body" or "subject". Defaults to "body".

        Raises:
            ValueError: If neither file_path nor texts is provided, or if field is invalid.
        """
        if file_path is None and texts is None:
            raise ValueError("Either file_path or texts must be provided.")
        if file_path is not None:
            if field not in ["body", "subject"]:
                raise ValueError("field must be either 'body' or 'subject'.")
            self.file_path = file_path
            self.field = field
            self.emails = pd.read_json(self.file_path)
            self.texts = self._extract_texts()
        else:
            self.texts = texts
            self.field = field

        self.nlp = spacy.load("en_core_web_sm")

    def _extract_texts(self) -> List[str]:
        """
        Extract the specified field (body or subject) from all email objects in the JSON file.

        Returns:
            List[str]: A list of extracted text strings.
        """
        texts = []
        for idx, row in self.emails.iterrows():
            email_chain = row.get("email_chain")
            if email_chain and isinstance(email_chain, list):
                email = email_chain[0]
                if (self.field in email and isinstance(email[self.field], str) and email[self.field].strip()):
                    texts.append(email[self.field])
        return texts

    def _get_parse_tree(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Generate a dependency parse representation of the text as a list of edges.

        Each edge is represented as a tuple:
            (token text, head token text, dependency label)

        Args:
            text (str): A text string.

        Returns:
            List[Tuple[str, str, str]]: A list of dependency edges for the given text.
        """
        doc = self.nlp(text)
        edges = [(token.text, token.head.text, token.dep_) for token in doc]
        return edges

    @staticmethod
    def tree_edit_dist(tree1: List[Tuple[str, str, str]], tree2: List[Tuple[str, str, str]]) -> float:
        """
        Placeholder for a tree edit distance function.

        Currently, this function returns the absolute difference in the number of edges between
        the two parse trees. For a robust measure, consider using an implementation of the Zhang-Shasha
        algorithm or another tree edit distance algorithm.

        Args:
            tree1 (List[Tuple[str, str, str]]): Parse tree of text 1.
            tree2 (List[Tuple[str, str, str]]): Parse tree of text 2.

        Returns:
            float: The computed tree edit distance.
        """
        return abs(len(tree1) - len(tree2))

    def compute_average_distance(self) -> float:
        """
        Computes the average pairwise parse-tree (tree edit) distance across all texts.

        Returns:
            float: The average tree edit distance, representing the syntactic diversity.
                   Returns 0.0 if fewer than 2 non-empty texts are available.
        """
        parse_trees = []
        for text in self.texts:
            if text.strip():
                tree = self._get_parse_tree(text)
                parse_trees.append(tree)

        n = len(parse_trees)
        if n < 2:
            return 0.0

        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                d = self.tree_edit_dist(parse_trees[i], parse_trees[j])
                distances.append(d)

        if not distances:
            return 0.0
        return sum(distances) / len(distances)

    def compute_overall_syntactic_diversity(self) -> float:
        """
        Computes the overall syntactic diversity using the available texts.

        Returns:
            float: The average syntactic diversity (average pairwise tree edit distance).
        """
        return self.compute_average_distance()

    def add_text(self, text: str) -> None:
        """
        Adds a new text to the collection.

        Args:
            text (str): The text to add.
        """
        self.texts.append(text)
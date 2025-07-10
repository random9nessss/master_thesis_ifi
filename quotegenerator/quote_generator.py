import os
import random
import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz

from config.logger import CustomLogger

class FreightQuoteEngine:

    def __init__(self, distance_matrix_path: str):
        """
        Initialize the Freight Quote Engine with a precomputed distance matrix.

        Parameters:
            distance_matrix_path (str): The file path to an Excel file containing the distance matrix.
                                          The Excel file should have port names as both its index and columns.
        """
        self._distance_matrix = pd.read_excel(distance_matrix_path, index_col=0)

        self.logger = CustomLogger(name="FreightQuoteEngineLogger")
        self.logger.ok("FreightQuoteEngine initialized")

    def _match_port(self, port: str) -> str:
        """
        Match the provided port name to one in the distance matrix using fuzzy matching.

        If the exact port name is not found, the function attempts to match it to an existing name
        in the distance matrix with a score of at least 80. If a good match is found, that matched name
        is returned; otherwise, the original name is returned (which may trigger a proxy distance).

        Parameters:
            port (str): The port name to match.

        Returns:
            str: The matched port name.
        """
        if port in self._distance_matrix.index:
            return port

        # -------------------------------------------------------------------
        # Fuzzy Matching
        # -------------------------------------------------------------------
        match = process.extractOne(port, self._distance_matrix.index, scorer=fuzz.ratio)
        if match and match[1] >= 80:
            return match[0]
        return port

    def _compute_rate(self, origin: str, destination: str) -> float:
        """
        Compute a shipping rate between two ports based on a base rate and a distance-based premium.

        The function looks up the distance in the precomputed distance matrix. If one of the ports is missing,
        it samples a random proxy distance from the nonzero entries in the matrix. A random variability factor
        (between 3 and 6) is applied to the distance premium.

        Parameters:
            origin (str): The origin port name (must be matched via fuzzy matching).
            destination (str): The destination port name (must be matched via fuzzy matching).

        Returns:
            float: The computed rate.
        """
        if (origin in self._distance_matrix.index) and (destination in self._distance_matrix.columns):
            dist_km = self._distance_matrix.loc[origin, destination]

            if isinstance(dist_km, pd.Series):
                dist_km = dist_km.mean()

        else:
            possible_distances = self._distance_matrix.values.flatten()
            possible_distances = possible_distances[possible_distances > 0]

            if len(possible_distances) > 0:
                dist_km = np.random.choice(possible_distances)
            else:
                dist_km = 1000

        # -------------------------------------------------------------------
        # Distance based freight premium
        # -------------------------------------------------------------------
        base_rate = 30
        factor = random.uniform(3, 6)
        additional = (dist_km / 1000) * factor
        return base_rate + additional

    def generate_rate(self, loadport: str,
                      dischargeport: str,
                      currency: str = None,
                      parity_ports: list = None,
                      gossip_prob: float = 0.10):
        """
        Generate a shipping rate based on the distance between ports, with optional parity quotes.

        This function performs the following steps:
          1. Uses fuzzy matching (via RapidFuzz) to match the provided port names to those in the distance matrix.
          2. Looks up the distance (in kilometers) between the load and discharge ports; if not found, it samples
             a proxy distance from the available nonzero distances.
          3. Computes a shipping rate as a base rate (30) plus an increment proportional to the distance,
             applying a random variability factor.
          4. If the optional parameter `parity_ports` is provided (a list of alternative destination ports), the engine:
                - Computes a numeric base quote for loadport → dischargeport.
                - Computes numeric quotes for each parity port (loadport → parity_port) and returns the delta
                  relative to the base quote.
                - Returns a list of tuples in the form:
                  (loadport, destination, currency, rate) for the base quote and
                  (loadport, parity_port, currency, delta) for each parity quote.
          5. If `parity_ports` is not provided and if `gossip_quote` is True, there is an 80% chance the function
             returns a descriptive textual quote (e.g., "mid 30s") instead of a numeric rate.
          6. If no parity quotes are requested and gossip_quote is not triggered, the function returns a numeric rate.
             A random choice between an integer and a float representation is made.
          7. If a valid currency symbol is provided (allowed: "$", "USD", "EUR", "AUD", "€"), it is prefixed to the result.

        Parameters:
            loadport (str):                 The name of the load port.
            dischargeport (str):            The name of the discharge port.
            currency (str, optional):       A valid currency symbol ("$", "USD", "EUR", "AUD", or "€") to prefix the rate.
            parity_ports (list, optional):  A list of alternative destination ports for parity quotes.
                                            If provided, numeric base and parity quotes are returned. Defaults to None.
            gossip_prob (float, optional)   Probabilty that a gossip quote is generated.

        Returns:
            If `parity_ports` is not provided:
                Union[str, float, int]: The generated shipping rate, either as a numeric value (with or without
                a currency prefix) or as a descriptive string (if gossip_quote is True and triggered).

            If `parity_ports` is provided:
                list of tuples: Each tuple is of the form (loadport, destination, currency, value), where the first
                tuple contains the base quote for (loadport, dischargeport) and subsequent tuples contain the delta
                (parity quote adjustment) for each alternative port.
                For example:
                    [(Rotterdam, Singapore, $, 80), (Rotterdam, Onsan, $, 2), (Rotterdam, Tokyo, $, -4)]
        """
        allowed_currencies = {"$", "USD", "EUR", "AUD", "€"}

        if currency is None:
            currency = random.choice(list(allowed_currencies))

        if currency and currency not in allowed_currencies:
            raise ValueError("Unsupported currency. Allowed values are: " + ", ".join(allowed_currencies))

        matched_loadport = self._match_port(loadport)
        matched_dischargeport = self._match_port(dischargeport)

        # -------------------------------------------------------------------
        # Parity Ports
        # -------------------------------------------------------------------
        if parity_ports is not None and len(parity_ports) > 0:
            base_rate = int(round(self._compute_rate(matched_loadport, matched_dischargeport)))
            results = []
            results.append((matched_loadport, matched_dischargeport, currency if currency else "", base_rate))

            for alt in parity_ports:
                matched_alt = self._match_port(alt)
                alt_rate = int(round(self._compute_rate(matched_loadport, matched_alt)))
                delta = alt_rate - base_rate
                results.append((matched_loadport, matched_alt, currency if currency else "", delta))
            return results

        rate = self._compute_rate(matched_loadport, matched_dischargeport)

        self.logger.info(f"Generated freight quote")

        # -------------------------------------------------------------------
        # Gossip Rate Formats
        # -------------------------------------------------------------------
        if random.random() <= gossip_prob:
            integer_rate = int(round(rate))
            decade = (integer_rate // 10) * 10
            remainder = integer_rate - decade

            if remainder <= 3:
                descriptor = "low"

            elif remainder <= 6:
                descriptor = "mid"

            else:
                descriptor = "high"

            descriptive_rate = f"{descriptor} {decade}s"
            return descriptive_rate

        # -------------------------------------------------------------------
        # Numerical Formatting
        # -------------------------------------------------------------------
        if random.choice(["int", "float"]) == "int":
            numeric_rate = int(round(rate))
            return f"{currency}{numeric_rate}" if currency else numeric_rate

        else:
            formatted_rate = f"{rate:.2f}"
            return f"{currency}{formatted_rate}" if currency else float(formatted_rate)
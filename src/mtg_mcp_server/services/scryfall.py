"""Scryfall API client for card search, lookup, and rulings.

Wraps Scryfall's REST API (https://scryfall.com/docs/api) with rate limiting,
retries, and Pydantic model parsing. All responses are cached via TTLCache.
"""

from __future__ import annotations

import structlog
from cachetools import TTLCache

from mtg_mcp_server.services.base import DEFAULT_USER_AGENT, BaseClient, ServiceError
from mtg_mcp_server.services.cache import _method_key, async_cached
from mtg_mcp_server.types import Card, CardPrices, CardSearchResult, Ruling, SetInfo

log = structlog.get_logger(service="scryfall")


class ScryfallError(ServiceError):
    """Scryfall API error."""


class CardNotFoundError(ScryfallError):
    """Card was not found on Scryfall."""


class ScryfallClient(BaseClient):
    """Async client for the Scryfall REST API.

    Requires ``User-Agent`` and ``Accept`` headers on every request (set by
    BaseClient). Rate limited to ~10 req/sec per Scryfall's guidelines.

    Args:
        base_url: Scryfall API base URL.
        rate_limit_rps: Max requests per second.
        user_agent: User-Agent header value.
    """

    # Class-level caches shared across all instances.
    # Card data rarely changes (24h TTL); searches rotate faster (1h TTL).
    _card_name_cache: TTLCache = TTLCache(maxsize=500, ttl=86400)  # 24h
    _card_id_cache: TTLCache = TTLCache(maxsize=500, ttl=86400)  # 24h
    _search_cache: TTLCache = TTLCache(maxsize=100, ttl=3600)  # 1h
    _rulings_cache: TTLCache = TTLCache(maxsize=200, ttl=86400)  # 24h
    _sets_cache: TTLCache = TTLCache(maxsize=50, ttl=86400)  # 24h
    _price_cache: TTLCache = TTLCache(maxsize=500, ttl=86400)  # 24h, prices update daily

    def __init__(
        self,
        base_url: str = "https://api.scryfall.com",
        rate_limit_rps: float = 10.0,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        super().__init__(
            base_url=base_url,
            rate_limit_rps=rate_limit_rps,
            user_agent=user_agent,
        )

    @async_cached(_card_name_cache, key=_method_key)
    async def get_card_by_name(self, name: str, *, fuzzy: bool = False) -> Card:
        """Look up a card by exact or fuzzy name.

        Args:
            name: Card name to search for.
            fuzzy: If True, use Scryfall's fuzzy matching instead of exact.

        Returns:
            Parsed Card model. When the returned printing has no USD price
            (new or digital-only printings), prices come from the cheapest
            paper printing instead — see :meth:`with_paper_price`.

        Raises:
            CardNotFoundError: If no card matches the name.
            ScryfallError: On other API errors.
        """
        param_key = "fuzzy" if fuzzy else "exact"
        try:
            response = await self._get("/cards/named", params={param_key: name})
        except ServiceError as exc:
            if exc.status_code == 404:
                raise CardNotFoundError(f"Card not found: '{name}'", status_code=404) from exc
            raise ScryfallError(exc.message, status_code=exc.status_code) from exc
        return await self.with_paper_price(Card.model_validate(response.json()))

    @async_cached(_price_cache, key=_method_key)
    async def get_paper_prices(self, name: str) -> CardPrices | None:
        """Prices of the cheapest printing of a card that has a USD price.

        Scryfall's default printing of a card is often its newest one, which
        has no market price yet (e.g. a just-released Commander product), so
        its ``prices.usd`` is null even for a $1 card like Sol Ring.

        Args:
            name: Exact card name.

        Returns:
            The cheapest USD-priced printing's prices, or None when no
            printing has a USD price (digital-only cards).

        Raises:
            ScryfallError: On API errors other than "no match".
        """
        exact = name.replace('"', "")
        try:
            result = await self.search_cards(f'!"{exact}" usd>0 prefer:usd-low')
        except CardNotFoundError:
            return None
        for card in result.data:
            if card.prices.usd is not None:
                return card.prices
        return None

    async def with_paper_price(self, card: Card) -> Card:
        """Return ``card`` with a USD price filled from its cheapest paper printing.

        Cards that already have a USD price are returned unchanged. Lookup
        failures are logged and leave the card unchanged — a missing price
        never fails the caller.
        """
        if card.prices.usd is not None:
            return card
        try:
            prices = await self.get_paper_prices(card.name)
        except ScryfallError as exc:
            log.warning("paper_price.failed", card=card.name, error=str(exc))
            return card
        if prices is None:
            return card
        merged = CardPrices(
            usd=prices.usd,
            usd_foil=prices.usd_foil if prices.usd_foil is not None else card.prices.usd_foil,
            eur=card.prices.eur if card.prices.eur is not None else prices.eur,
        )
        return card.model_copy(update={"prices": merged})

    @async_cached(_search_cache, key=_method_key)
    async def search_cards(self, query: str, page: int = 1) -> CardSearchResult:
        """Search for cards using Scryfall syntax.

        Args:
            query: Scryfall search query (e.g. ``"f:commander id:sultai"``).
            page: Results page number (1-indexed).

        Returns:
            Paginated search results with card list and ``has_more`` flag.

        Raises:
            CardNotFoundError: If no cards match the query (404).
            ScryfallError: On other API errors.
        """
        try:
            response = await self._get("/cards/search", params={"q": query, "page": str(page)})
        except ServiceError as exc:
            if exc.status_code == 404:
                raise CardNotFoundError(
                    f"No cards found for query: '{query}'", status_code=404
                ) from exc
            raise ScryfallError(exc.message, status_code=exc.status_code) from exc
        return CardSearchResult.model_validate(response.json())

    @async_cached(_card_id_cache, key=_method_key)
    async def get_card_by_id(self, scryfall_id: str) -> Card:
        """Look up a card by Scryfall UUID.

        Args:
            scryfall_id: Scryfall card UUID.

        Returns:
            Parsed Card model.

        Raises:
            CardNotFoundError: If the UUID does not exist.
            ScryfallError: On other API errors.
        """
        try:
            response = await self._get(f"/cards/{scryfall_id}")
        except ServiceError as exc:
            if exc.status_code == 404:
                raise CardNotFoundError(
                    f"Card not found: '{scryfall_id}'", status_code=404
                ) from exc
            raise ScryfallError(exc.message, status_code=exc.status_code) from exc
        return Card.model_validate(response.json())

    @async_cached(_rulings_cache, key=_method_key)
    async def get_rulings(self, scryfall_id: str) -> list[Ruling]:
        """Get official rulings for a card by Scryfall UUID.

        Args:
            scryfall_id: Scryfall card UUID.

        Returns:
            List of rulings with source, date, and comment text.

        Raises:
            ScryfallError: On API errors.
        """
        try:
            response = await self._get(f"/cards/{scryfall_id}/rulings")
        except ServiceError as exc:
            raise ScryfallError(exc.message, status_code=exc.status_code) from exc
        data = response.json()
        return [Ruling.model_validate(r) for r in data["data"]]

    @async_cached(_sets_cache, key=_method_key)
    async def get_sets(self) -> list[SetInfo]:
        """List all Magic sets.

        Returns:
            List of SetInfo models.

        Raises:
            ScryfallError: On API errors.
        """
        try:
            response = await self._get("/sets")
        except ServiceError as exc:
            raise ScryfallError(exc.message, status_code=exc.status_code) from exc
        return [SetInfo.model_validate(s) for s in response.json()["data"]]

    @async_cached(_sets_cache, key=_method_key)
    async def get_set(self, set_code: str) -> SetInfo:
        """Get set metadata by code.

        Args:
            set_code: Set code (e.g. "dom", "mh2").

        Returns:
            Parsed SetInfo model.

        Raises:
            CardNotFoundError: If the set code doesn't exist.
            ScryfallError: On other API errors.
        """
        try:
            response = await self._get(f"/sets/{set_code}")
        except ServiceError as exc:
            if exc.status_code == 404:
                raise CardNotFoundError(f"Set not found: '{set_code}'", status_code=404) from exc
            raise ScryfallError(exc.message, status_code=exc.status_code) from exc
        return SetInfo.model_validate(response.json())

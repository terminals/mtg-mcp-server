"""17Lands API client for draft card ratings and archetype statistics.

Wraps 17Lands' undocumented but widely-used REST endpoints for card win rates
and color pair performance data. Rate limited aggressively (1 req/sec max).
"""

from __future__ import annotations

from cachetools import TTLCache

from mtg_mcp_server.services.base import DEFAULT_USER_AGENT, BaseClient, ServiceError
from mtg_mcp_server.services.cache import _method_key, async_cached
from mtg_mcp_server.types import ArchetypeRating, DraftCardRating


class SeventeenLandsError(ServiceError):
    """17Lands API error."""


# Default date range for card_ratings — the API only returns game data for past
# formats when start_date/end_date are provided.  A broad static range covers
# every set 17Lands has ever tracked without making the tests date-dependent.
_DEFAULT_START_DATE = "2020-01-01"
_DEFAULT_END_DATE = "2099-12-31"

_CARD_STATS_UNAVAILABLE = (
    "17Lands returned no per-card win rates for {set_code}. 17Lands no longer "
    "publishes card stats through its public API (they now say the data is for "
    "use on 17lands.com only), or the set is too new to have data. "
    "Color-pair win rates still work via draft_archetype_stats; for card stats, "
    "open https://www.17lands.com/card_data in a browser."
)


class SeventeenLandsClient(BaseClient):
    """Async client for the 17Lands draft data API.

    Rate limit is aggressive (1 req/sec max). All responses are cached with
    a 4-hour TTL to minimize API pressure.

    Args:
        base_url: 17Lands base URL.
        rate_limit_rps: Max requests per second (default 1.0).
        user_agent: User-Agent header value.
    """

    _card_ratings_cache: TTLCache = TTLCache(maxsize=20, ttl=14400)  # 4h
    _color_ratings_cache: TTLCache = TTLCache(maxsize=20, ttl=14400)  # 4h

    def __init__(
        self,
        base_url: str = "https://www.17lands.com",
        rate_limit_rps: float = 1.0,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        super().__init__(
            base_url=base_url,
            rate_limit_rps=rate_limit_rps,
            user_agent=user_agent,
        )

    @async_cached(_card_ratings_cache, key=_method_key)
    async def card_ratings(
        self,
        set_code: str,
        event_type: str = "PremierDraft",
    ) -> list[DraftCardRating]:
        """Get card performance data for a set.

        A broad default date range is always sent because the 17Lands API only
        returns game data for past formats when ``start_date``/``end_date`` are
        present. Without them, only the current format returns non-zero counts.

        Args:
            set_code: Set code (e.g. ``"LCI"``, ``"MKM"``).
            event_type: Draft format (e.g. ``"PremierDraft"``, ``"TradDraft"``).

        Returns:
            List of card ratings with win rates, pick rates, etc.

        Raises:
            SeventeenLandsError: On API errors.
        """
        try:
            response = await self._get(
                "/card_ratings/data",
                params={
                    "expansion": set_code.upper(),
                    "event_type": event_type,
                    "start_date": _DEFAULT_START_DATE,
                    "end_date": _DEFAULT_END_DATE,
                },
            )
        except ServiceError as exc:
            raise SeventeenLandsError(exc.message, status_code=exc.status_code) from exc
        ratings = [DraftCardRating.model_validate(item) for item in response.json()]
        if ratings and all(r.ever_drawn_win_rate is None for r in ratings):
            # Since 2026 this endpoint still lists every card but returns no
            # win rates for any set. 17Lands moved per-card stats to a site-only
            # API whose responses say the data is for use on 17lands.com only,
            # so this server does not call it. Fail loudly instead of returning
            # a table of N/A values that looks like real data.
            raise SeventeenLandsError(_CARD_STATS_UNAVAILABLE.format(set_code=set_code.upper()))
        return ratings

    @async_cached(_color_ratings_cache, key=_method_key)
    async def color_ratings(
        self,
        set_code: str,
        start_date: str,
        end_date: str,
        event_type: str = "PremierDraft",
    ) -> list[ArchetypeRating]:
        """Get archetype win rates by color pair for a set.

        Args:
            set_code: Set code (e.g. ``"LCI"``, ``"MKM"``).
            start_date: Start date in YYYY-MM-DD format (required by API).
            end_date: End date in YYYY-MM-DD format (required by API).
            event_type: Draft format (e.g. ``"PremierDraft"``, ``"TradDraft"``).

        Returns:
            List of archetype ratings with wins, games, and derived win rates.

        Raises:
            SeventeenLandsError: On API errors (including 400 if dates are missing).
        """
        try:
            response = await self._get(
                "/color_ratings/data",
                params={
                    "expansion": set_code.upper(),
                    "event_type": event_type,
                    "start_date": start_date,
                    "end_date": end_date,
                },
            )
        except ServiceError as exc:
            raise SeventeenLandsError(exc.message, status_code=exc.status_code) from exc
        return [ArchetypeRating.model_validate(item) for item in response.json()]

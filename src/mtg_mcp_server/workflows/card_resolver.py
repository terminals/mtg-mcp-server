"""Card resolver — bulk-data-first card resolution with Scryfall fallback."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

from mtg_mcp_server.services.base import ServiceError
from mtg_mcp_server.types import Card

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mtg_mcp_server.services.scryfall import ScryfallClient
    from mtg_mcp_server.services.scryfall_bulk import ScryfallBulkClient

log = structlog.get_logger(service="workflow.card_resolver")

# Cap concurrent Scryfall fallback lookups to avoid overwhelming the
# connection pool (mirrors deck_analysis's bound).
_MAX_FALLBACK_CONCURRENCY = 10

# Most Scryfall price lookups one call will make for unpriced bulk cards
# (each is one rate-limited API request).
_MAX_PRICE_LOOKUPS = 60


async def resolve_card(
    name: str,
    *,
    bulk: ScryfallBulkClient | None,
    scryfall: ScryfallClient,
) -> Card:
    """Resolve a card using bulk data first, falling back to Scryfall.

    When bulk data is available, tries it first for rate-limit-free lookup.
    Falls back to Scryfall if the card is not found in bulk data or if
    bulk data is disabled.

    Args:
        name: Card name to look up.
        bulk: Initialized ScryfallBulkClient, or None if disabled.
        scryfall: Initialized ScryfallClient (always available).

    Returns:
        A Card object (from bulk data or Scryfall — same type either way).

    Raises:
        CardNotFoundError: If the card is not found in any source.
    """
    if bulk is not None:
        try:
            card = await bulk.get_card(name)
        except Exception:
            log.warning("resolve_card.bulk_error", name=name, exc_info=True)
            card = None

        if card is not None:
            log.debug("resolve_card.bulk_hit", name=name)
            return card
        log.debug("resolve_card.bulk_miss", name=name)

    log.debug("resolve_card.scryfall_lookup", name=name)
    return await scryfall.get_card_by_name(name)


async def fill_missing_prices(
    cards: Iterable[Card],
    *,
    scryfall: ScryfallClient,
    max_lookups: int = _MAX_PRICE_LOOKUPS,
) -> dict[str, Card]:
    """Look up USD prices for cards whose bulk printing has none.

    Scryfall's Oracle bulk data carries one printing per card, often the
    newest; a just-released printing has no market price yet, so staples
    such as Sol Ring come back with ``prices.usd = None``. This asks
    Scryfall for the cheapest priced paper printing of each such card
    (concurrency-capped, at most ``max_lookups`` requests).

    Args:
        cards: Cards to check (cards that already have a USD price are skipped).
        scryfall: Initialized ScryfallClient.
        max_lookups: Upper bound on price requests.

    Returns:
        Mapping of lowercase card name to the re-priced Card, only for cards
        that gained a USD price. Never raises.
    """
    unpriced: list[Card] = []
    seen: set[str] = set()
    for card in cards:
        key = card.name.lower()
        if card.prices.usd is None and key not in seen:
            seen.add(key)
            unpriced.append(card)
    if not unpriced:
        return {}
    if len(unpriced) > max_lookups:
        log.info("fill_missing_prices.capped", unpriced=len(unpriced), max_lookups=max_lookups)
        unpriced = unpriced[:max_lookups]

    sem = asyncio.Semaphore(_MAX_FALLBACK_CONCURRENCY)

    async def _price(card: Card) -> Card | None:
        async with sem:
            try:
                repriced = await scryfall.with_paper_price(card)
            except ServiceError as exc:
                log.warning("fill_missing_prices.failed", card=card.name, error=str(exc))
                return None
        if isinstance(repriced, Card) and repriced.prices.usd is not None:
            return repriced
        return None

    results = await asyncio.gather(*(_price(card) for card in unpriced))
    filled = {card.name.lower(): card for card in results if card is not None}
    log.debug("fill_missing_prices.done", unpriced=len(unpriced), filled=len(filled))
    return filled


async def resolve_cards(
    names: list[str],
    *,
    bulk: ScryfallBulkClient | None,
    scryfall: ScryfallClient,
    fill_prices: bool = False,
) -> dict[str, Card | None]:
    """Resolve many cards by name: bulk-data-first with Scryfall fallback.

    Batch counterpart to :func:`resolve_card` with the same degradation
    semantics, but using a single ``bulk.get_cards`` call for the bulk
    stage: a bulk error or bulk miss for a card falls back to Scryfall for
    that card (concurrency-capped). A card not found anywhere maps to
    ``None`` — this function never raises for individual cards, matching
    ``bulk.get_cards``' shape.

    Args:
        names: Card names to look up.
        bulk: Initialized ScryfallBulkClient, or None if disabled.
        scryfall: Initialized ScryfallClient (fallback source).
        fill_prices: Also look up a USD price for bulk cards that lack one
            (see :func:`fill_missing_prices`). Use it when prices matter.

    Returns:
        Dict mapping each input name (original casing preserved) to its
        Card, or None if not found in any source.
    """
    # Dedupe input names (decklists repeat cards) so the bulk batch and
    # each fallback path fetch every distinct name at most once.
    unique_names = list(dict.fromkeys(names))
    resolved: dict[str, Card | None] = {name: None for name in unique_names}
    missing: list[str] = unique_names

    if bulk is not None:
        try:
            bulk_map = await bulk.get_cards(unique_names)
        except Exception:
            log.warning("resolve_cards.bulk_error", count=len(unique_names), exc_info=True)
        else:
            missing = []
            for name in unique_names:
                card = bulk_map.get(name)
                if card is not None:
                    resolved[name] = card
                else:
                    missing.append(name)
            if missing:
                log.debug(
                    "resolve_cards.bulk_miss",
                    total=len(unique_names),
                    missing=len(missing),
                )

    if fill_prices:
        bulk_hits = [card for name, card in resolved.items() if card is not None]
        repriced = await fill_missing_prices(bulk_hits, scryfall=scryfall)
        for name, card in resolved.items():
            if card is not None and card.name.lower() in repriced:
                resolved[name] = repriced[card.name.lower()]

    if not missing:
        return resolved

    # Scryfall's get_card_by_name already fills paper prices itself.
    log.debug("resolve_cards.scryfall_fallback", count=len(missing))
    sem = asyncio.Semaphore(_MAX_FALLBACK_CONCURRENCY)

    async def _fetch(name: str) -> tuple[str, Card | None]:
        async with sem:
            try:
                card = await scryfall.get_card_by_name(name)
            except ServiceError as exc:
                log.warning(
                    "resolve_cards.scryfall_failed",
                    card=name,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                return name, None
            return name, card

    fallback = await asyncio.gather(*(_fetch(name) for name in missing))
    for name, card in fallback:
        resolved[name] = card
    return resolved

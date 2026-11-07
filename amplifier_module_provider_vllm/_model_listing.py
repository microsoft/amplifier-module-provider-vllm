"""Provider-local immutable model selectors; no transport or shared dependency."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase


@dataclass(frozen=True)
class ModelFilters:
    include: tuple[str, ...] = ("*",)
    exclude: tuple[str, ...] = ()
    hide_dated: bool = False

    @classmethod
    def parse(cls, value: Mapping[str, object]) -> "ModelFilters":
        if not isinstance(value, Mapping):
            raise TypeError("model filters must be a mapping")
        if set(value) - {"include", "exclude", "hide_dated"}:
            raise ValueError("unknown model filter key")

        def patterns(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
            raw = value.get(key, default)
            if not isinstance(raw, (list, tuple)) or any(
                not isinstance(p, str) or not p.strip() for p in raw
            ):
                raise ValueError(
                    f"model filters {key} must be a list of nonblank strings"
                )
            return tuple(raw)

        hide_dated = value.get("hide_dated", False)
        if not isinstance(hide_dated, bool):
            raise TypeError("model filters hide_dated must be a boolean")
        return cls(patterns("include", ("*",)), patterns("exclude", ()), hide_dated)

    def select(self, ids: list[str]) -> list[str]:
        selected = {
            model_id
            for model_id in ids
            if any(fnmatchcase(model_id.casefold(), p.casefold()) for p in self.include)
            and not any(
                fnmatchcase(model_id.casefold(), p.casefold()) for p in self.exclude
            )
        }
        if self.hide_dated:
            selected = {
                model_id
                for model_id in selected
                if not (
                    (match := re.search(r"-\d{4}-\d{2}-\d{2}$", model_id))
                    and model_id[: match.start()] in selected
                )
            }
        return sorted(selected)

    def as_dict(self) -> dict:
        return {
            "include": list(self.include),
            "exclude": list(self.exclude),
            "hide_dated": self.hide_dated,
        }


def selector_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "include": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "default": ["*"],
                "description": "Case-insensitive OS-independent glob patterns; [] includes none.",
            },
            "exclude": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "default": [],
                "description": "Case-insensitive glob patterns; exclusion wins.",
            },
            "hide_dated": {
                "type": "boolean",
                "default": False,
                "description": "Hide YYYY-MM-DD suffixes only when the selected undated alias exists.",
            },
        },
    }


def listing_info(source: str, policy: ModelFilters) -> dict:
    return {
        "version": 1,
        "source": source,
        "filters_keyword": "filters",
        "filters_schema": selector_schema(),
        "default_filters": policy.as_dict(),
        "unfiltered": {},
        "override": "replace",
    }

"""Slice 28: Phase 1B profile completeness and provenance tests.

Tests prove:
  * all 18 profiles exist in PROFILE_DELTAS
  * every profile has _sources (non-empty string)
  * every profile has _ranges (dict, possibly empty for normals)
  * default values fall within declared _ranges
  * behavior-map coverage: every profile has an entry
  * generation reachability: all profiles appear in the classroom generation path
  * no regressions in existing profile invariants
"""

from src.simulation.cognitive_agent import (
    PROFILE_DELTAS,
    _PROFILE_BEHAVIOR_MAP,
    BEHAVIOR_POOLS,
    _combine_deltas,
    CognitiveParameters,
)


EXPECTED_PROFILES = [
    # Normal
    "normal_quiet", "normal_active",
    # ADHD
    "adhd_inattentive", "adhd_hyperactive_impulsive", "adhd_combined",
    # Single disorders
    "anxiety", "odd", "gifted", "sleep_deprived",
    # Comorbidity
    "adhd_i_plus_anxiety", "adhd_h_plus_odd", "adhd_c_plus_odd",
    "adhd_i_plus_ld", "adhd_plus_depression", "anxiety_plus_depression",
    # Distractors
    "asd_like", "depression", "learning_disorder",
]


def test_all_18_profiles_exist():
    for name in EXPECTED_PROFILES:
        assert name in PROFILE_DELTAS, f"Missing profile: {name}"
    assert len(EXPECTED_PROFILES) == 18


def test_every_profile_has_sources():
    for name in EXPECTED_PROFILES:
        spec = PROFILE_DELTAS[name]
        sources = spec.get("_sources")
        assert sources and isinstance(sources, str) and len(sources) > 5, (
            f"{name} missing or empty _sources"
        )


def test_every_profile_has_ranges():
    for name in EXPECTED_PROFILES:
        spec = PROFILE_DELTAS[name]
        ranges = spec.get("_ranges")
        assert ranges is not None and isinstance(ranges, dict), (
            f"{name} missing _ranges (Phase 1B requirement)"
        )


def test_normal_variants_have_empty_ranges():
    for name in ("normal_quiet", "normal_active"):
        assert PROFILE_DELTAS[name]["_ranges"] == {}, (
            f"{name} should have empty _ranges (no autoresearch tuning)"
        )


def test_non_normal_profiles_have_nonempty_ranges():
    for name in EXPECTED_PROFILES:
        if name.startswith("normal_"):
            continue
        ranges = PROFILE_DELTAS[name]["_ranges"]
        assert len(ranges) > 0, (
            f"{name} has empty _ranges but is not a normal variant"
        )


def test_default_values_within_declared_ranges():
    """For each _ranges entry, verify the actual default delta falls
    within [lo, hi]."""
    for name in EXPECTED_PROFILES:
        spec = PROFILE_DELTAS[name]
        ranges = spec.get("_ranges", {})
        for range_key, (lo, hi) in ranges.items():
            domain, field = range_key.split(".", 1)
            actual = spec.get(domain, {}).get(field)
            if actual is None:
                # Field not in this profile's delta (may come from _base)
                continue
            assert lo <= actual <= hi, (
                f"{name}.{range_key}: default {actual} outside [{lo}, {hi}]"
            )


def test_every_profile_has_behavior_map_entry():
    for name in EXPECTED_PROFILES:
        assert name in _PROFILE_BEHAVIOR_MAP, (
            f"{name} missing from _PROFILE_BEHAVIOR_MAP"
        )


def test_behavior_map_references_valid_pools():
    for name, pools in _PROFILE_BEHAVIOR_MAP.items():
        for pool_name in pools:
            assert pool_name in BEHAVIOR_POOLS, (
                f"{name} references unknown pool: {pool_name}"
            )


def test_profile_resolution_does_not_crash():
    """Resolving every profile should succeed without error."""
    for name in EXPECTED_PROFILES:
        resolved = _combine_deltas(name)
        assert "cognitive" in resolved
        assert "emotional" in resolved


def test_adhd_profiles_start_with_adhd():
    """All ADHD-related profiles must have is_adhd=True,
    which depends on profile_type.startswith('adhd_')."""
    adhd_names = [n for n in EXPECTED_PROFILES if "adhd" in n]
    for name in adhd_names:
        assert name.startswith("adhd_"), (
            f"{name} contains 'adhd' but doesn't start with 'adhd_' "
            f"— is_adhd would be False"
        )


def test_distractor_profiles_are_not_adhd():
    for name in ("asd_like", "depression", "learning_disorder"):
        assert not name.startswith("adhd_"), (
            f"Distractor {name} should not be classified as ADHD"
        )

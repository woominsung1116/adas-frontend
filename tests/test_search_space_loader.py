"""Tests for YAML → SearchSpace loader (Phase 4.5 bridge).

Covers:
  - Real .harness/student_parameter_ranges.yaml loads cleanly
  - Expected parameter names present, with correct kind/range/default
  - Int vs float distinction for base_cognitive fields
  - Live-simulator fallback for defaults
  - Midpoint fallback when no explicit or live default
  - Constraints preserved verbatim
  - Metadata surfaced
  - Unsupported sections recorded (not crashed)
  - Malformed entries raise InvalidParameterSpecError loudly
  - Public default factory returns usable SearchSpace
"""

from pathlib import Path

import pytest

from src.calibration import (
    InvalidParameterSpecError,
    LoadedSearchSpace,
    load_search_space,
    load_default_search_space,
    build_default_search_space,
    default_student_ranges_path,
    SearchSpace,
)
from src.calibration.search_space_loader import _normalize_range, _choose_default


# ==========================================================================
# Real harness file
# ==========================================================================


def test_default_path_exists():
    assert default_student_ranges_path().exists()


def test_load_default_returns_non_empty_space():
    loaded = load_default_search_space()
    assert isinstance(loaded, LoadedSearchSpace)
    assert len(loaded.space) > 0
    assert isinstance(loaded.space, SearchSpace)


def test_build_default_search_space_shortcut():
    space = build_default_search_space()
    assert isinstance(space, SearchSpace)
    assert len(space) > 0


def test_expected_parameter_names_present():
    loaded = load_default_search_space()
    names = loaded.space.names()
    # Known base cognitive parameters
    assert "base_cognitive.att_bandwidth" in names
    assert "base_cognitive.vision_r" in names
    assert "base_cognitive.retention" in names
    assert "base_cognitive.plan_consistency" in names
    assert "base_cognitive.impulse_override" in names
    # Base emotional
    assert "base_emotional.frustration" in names
    assert "base_emotional.shame" in names
    assert "base_emotional.trust_in_teacher" in names
    # ADHD-I delta (cognitive)
    assert "adhd_inattentive.cognitive.att_bandwidth" in names
    assert "adhd_inattentive.cognitive.retention" in names
    # ADHD-I delta (emotional)
    assert "adhd_inattentive.emotional.frustration" in names
    # ADHD-H delta — alias mapping from "adhd_hyperactive" to full name
    assert "adhd_hyperactive_impulsive.cognitive.impulse_override" in names
    assert "adhd_hyperactive_impulsive.emotional.anger" in names


def test_integer_fields_have_int_kind():
    loaded = load_default_search_space()
    specs_by_name = {s.name: s for s in loaded.space.specs}
    # att_bandwidth / vision_r / retention are int
    assert specs_by_name["base_cognitive.att_bandwidth"].kind == "int"
    assert specs_by_name["base_cognitive.vision_r"].kind == "int"
    assert specs_by_name["base_cognitive.retention"].kind == "int"
    # Non-int cognitive field is float
    assert specs_by_name["base_cognitive.plan_consistency"].kind == "float"


def test_int_spec_has_int_bounds_and_default():
    loaded = load_default_search_space()
    specs_by_name = {s.name: s for s in loaded.space.specs}
    s = specs_by_name["base_cognitive.att_bandwidth"]
    assert s.kind == "int"
    assert isinstance(s.lo, int)
    assert isinstance(s.hi, int)
    assert isinstance(s.default, int)
    assert s.lo == 2
    assert s.hi == 4
    assert s.default == 3  # explicit YAML value


def test_explicit_default_takes_precedence():
    """YAML explicit default wins over live/midpoint."""
    loaded = load_default_search_space()
    specs_by_name = {s.name: s for s in loaded.space.specs}
    # importance_trigger has explicit default=135.0
    s = specs_by_name["base_cognitive.importance_trigger"]
    assert s.default == pytest.approx(135.0)


def test_source_citations_preserved():
    loaded = load_default_search_space()
    assert "base_cognitive.att_bandwidth" in loaded.sources
    assert "Miller" in loaded.sources["base_cognitive.att_bandwidth"]
    # ParameterSpec itself also carries source
    specs_by_name = {s.name: s for s in loaded.space.specs}
    assert "Miller" in specs_by_name["base_cognitive.att_bandwidth"].source


def test_constraints_preserved_verbatim():
    loaded = load_default_search_space()
    assert len(loaded.constraints) >= 4
    rules = [c["rule"] for c in loaded.constraints]
    # Known constraint substrings from the real YAML
    assert any("att_bandwidth" in r and "impulse_override" in r for r in rules)
    # Constraint entries carry profile + rationale
    for c in loaded.constraints:
        assert "rule" in c
        assert "profile" in c or "rule" in c  # at least one identifier


def test_metadata_surfaced():
    loaded = load_default_search_space()
    assert "version" in loaded.metadata
    assert "philosophy" in loaded.metadata


def test_unsupported_sections_recorded():
    loaded = load_default_search_space()
    # These YAML sections exist but aren't wired into the applier yet
    expected = {
        "emotion_decay_rates",
        "event_reactivity",
        "emotion_to_cognition_coupling",
        "comorbidity_interactions",
    }
    assert expected.issubset(set(loaded.unsupported_sections))


# ==========================================================================
# Defaults from live simulator fallback
# ==========================================================================


def test_live_simulator_default_fallback(tmp_path):
    """A YAML entry without explicit default should fall back to the
    live simulator value (BASE_EMOTIONAL[field])."""
    from src.simulation import cognitive_agent as ca

    # Pick a field whose live value is known
    live_anxiety = ca.BASE_EMOTIONAL["anxiety"]

    fake_yaml = tmp_path / "live_fallback.yaml"
    fake_yaml.write_text(
        """
base_emotional:
  anxiety:
    range: [0.0, 1.0]
    # no explicit default
""",
        encoding="utf-8",
    )
    loaded = load_search_space(fake_yaml)
    spec = [s for s in loaded.space.specs if s.name == "base_emotional.anxiety"][0]
    assert spec.default == pytest.approx(live_anxiety)


def test_midpoint_fallback_when_no_live_value(tmp_path):
    """A YAML entry for a field not in the simulator should fall back
    to the midpoint of [lo, hi]."""
    fake_yaml = tmp_path / "midpoint.yaml"
    fake_yaml.write_text(
        """
base_emotional:
  definitely_not_a_real_field:
    range: [0.2, 0.8]
""",
        encoding="utf-8",
    )
    loaded = load_search_space(fake_yaml)
    spec = [s for s in loaded.space.specs][0]
    # Live lookup returns None → midpoint
    assert spec.default == pytest.approx(0.5)


# ==========================================================================
# Choice parameter support (Phase 4.5 corrective pass)
# ==========================================================================


def test_choice_param_base_section_parses(tmp_path):
    """A base_* entry with `kind: choice` + `choices: [...]` loads as
    a choice ParameterSpec."""
    fake_yaml = tmp_path / "choice_base.yaml"
    fake_yaml.write_text(
        """
base_emotional:
  mood_variant:
    kind: choice
    choices: ["calm", "anxious", "excited"]
    default: "calm"
    source: "synthetic test"
""",
        encoding="utf-8",
    )
    loaded = load_search_space(fake_yaml)
    spec = [s for s in loaded.space.specs if s.name == "base_emotional.mood_variant"][0]
    assert spec.kind == "choice"
    assert spec.choices == ["calm", "anxious", "excited"]
    assert spec.default == "calm"
    assert spec.source == "synthetic test"


def test_choice_param_default_falls_back_to_first_choice(tmp_path):
    """Omitting `default:` on a choice entry → first choice is used."""
    fake_yaml = tmp_path / "choice_default.yaml"
    fake_yaml.write_text(
        """
base_emotional:
  variant:
    kind: choice
    choices: ["alpha", "beta"]
""",
        encoding="utf-8",
    )
    loaded = load_search_space(fake_yaml)
    spec = loaded.space.specs[0]
    assert spec.kind == "choice"
    assert spec.default == "alpha"


def test_choice_param_missing_choices_raises(tmp_path):
    fake_yaml = tmp_path / "choice_bad1.yaml"
    fake_yaml.write_text(
        """
base_emotional:
  variant:
    kind: choice
    # no choices key
""",
        encoding="utf-8",
    )
    with pytest.raises(InvalidParameterSpecError, match="requires 'choices' list"):
        load_search_space(fake_yaml)


def test_choice_param_empty_choices_raises(tmp_path):
    fake_yaml = tmp_path / "choice_bad2.yaml"
    fake_yaml.write_text(
        """
base_emotional:
  variant:
    kind: choice
    choices: []
""",
        encoding="utf-8",
    )
    with pytest.raises(InvalidParameterSpecError, match="choices' list is empty"):
        load_search_space(fake_yaml)


def test_choice_param_choices_not_list_raises(tmp_path):
    fake_yaml = tmp_path / "choice_bad3.yaml"
    fake_yaml.write_text(
        """
base_emotional:
  variant:
    kind: choice
    choices: "not a list"
""",
        encoding="utf-8",
    )
    with pytest.raises(InvalidParameterSpecError, match="'choices' must be a list"):
        load_search_space(fake_yaml)


def test_choice_param_default_not_in_choices_raises(tmp_path):
    fake_yaml = tmp_path / "choice_bad4.yaml"
    fake_yaml.write_text(
        """
base_emotional:
  variant:
    kind: choice
    choices: ["a", "b", "c"]
    default: "z"
""",
        encoding="utf-8",
    )
    with pytest.raises(InvalidParameterSpecError, match="not in choices"):
        load_search_space(fake_yaml)


def test_choice_param_in_profile_delta_section(tmp_path):
    """Choice-kind entries also work inside `<profile>_deltas`."""
    fake_yaml = tmp_path / "choice_delta.yaml"
    fake_yaml.write_text(
        """
adhd_inattentive_deltas:
  emotional:
    stress_variant:
      kind: choice
      choices: ["low", "med", "high"]
      default: "med"
""",
        encoding="utf-8",
    )
    loaded = load_search_space(fake_yaml)
    spec = [s for s in loaded.space.specs][0]
    assert spec.name == "adhd_inattentive.emotional.stress_variant"
    assert spec.kind == "choice"
    assert spec.default == "med"


def test_choice_param_random_proposer_samples(tmp_path):
    """RandomProposer must be able to sample a loaded choice spec."""
    import random
    from src.calibration.proposer import RandomProposer

    fake_yaml = tmp_path / "choice_sample.yaml"
    fake_yaml.write_text(
        """
base_emotional:
  variant:
    kind: choice
    choices: ["a", "b", "c"]
""",
        encoding="utf-8",
    )
    loaded = load_search_space(fake_yaml)
    proposer = RandomProposer(loaded.space, seed=42)
    sampled_values = set()
    for _ in range(50):
        cfg = proposer.propose([])
        val = cfg["base_emotional.variant"]
        assert val in ["a", "b", "c"]
        sampled_values.add(val)
    # With 50 draws from 3 choices, all 3 should appear
    assert sampled_values == {"a", "b", "c"}


def test_choice_param_clip_coerces_invalid_to_first():
    """SearchSpace.clip_config maps an unknown choice value to choices[0]."""
    from src.calibration.proposer import ParameterSpec, SearchSpace

    space = SearchSpace([
        ParameterSpec("x", kind="choice", choices=["a", "b", "c"], default="a"),
    ])
    clipped = space.clip_config({"x": "z"})
    assert clipped["x"] == "a"


# ==========================================================================
# Malformed YAML entries — fail loudly
# ==========================================================================


def test_malformed_missing_range_raises(tmp_path):
    fake_yaml = tmp_path / "bad1.yaml"
    fake_yaml.write_text(
        """
base_emotional:
  shame:
    default: 0.1
    # missing range
""",
        encoding="utf-8",
    )
    with pytest.raises(InvalidParameterSpecError, match="missing 'range'"):
        load_search_space(fake_yaml)


def test_malformed_inverted_range_raises(tmp_path):
    fake_yaml = tmp_path / "bad2.yaml"
    fake_yaml.write_text(
        """
base_emotional:
  shame:
    range: [0.8, 0.2]
""",
        encoding="utf-8",
    )
    with pytest.raises(InvalidParameterSpecError, match="lo.*> hi"):
        load_search_space(fake_yaml)


def test_malformed_range_shape_raises(tmp_path):
    fake_yaml = tmp_path / "bad3.yaml"
    fake_yaml.write_text(
        """
base_emotional:
  shame:
    range: "not a list"
""",
        encoding="utf-8",
    )
    with pytest.raises(InvalidParameterSpecError):
        load_search_space(fake_yaml)


def test_unknown_delta_subsection_raises(tmp_path):
    fake_yaml = tmp_path / "bad4.yaml"
    fake_yaml.write_text(
        """
adhd_inattentive_deltas:
  weird_subsection:
    frustration:
      range: [0.1, 0.3]
""",
        encoding="utf-8",
    )
    with pytest.raises(InvalidParameterSpecError, match="unknown subsection"):
        load_search_space(fake_yaml)


def test_malformed_constraint_missing_rule_raises(tmp_path):
    fake_yaml = tmp_path / "bad5.yaml"
    fake_yaml.write_text(
        """
constraints:
  - profile: "adhd_inattentive"
    rationale: "no rule field"
""",
        encoding="utf-8",
    )
    with pytest.raises(InvalidParameterSpecError, match="missing 'rule'"):
        load_search_space(fake_yaml)


def test_constraints_not_list_raises(tmp_path):
    fake_yaml = tmp_path / "bad6.yaml"
    fake_yaml.write_text(
        """
constraints:
  some_key: "not a list"
""",
        encoding="utf-8",
    )
    with pytest.raises(InvalidParameterSpecError, match="must be a list"):
        load_search_space(fake_yaml)


def test_non_dict_yaml_root_raises(tmp_path):
    fake_yaml = tmp_path / "bad7.yaml"
    fake_yaml.write_text(
        """
- just
- a
- list
""",
        encoding="utf-8",
    )
    with pytest.raises(InvalidParameterSpecError, match="root must be a mapping"):
        load_search_space(fake_yaml)


# ==========================================================================
# Helper unit tests
# ==========================================================================


def test_normalize_range_list():
    assert _normalize_range([0.1, 0.5]) == (0.1, 0.5)


def test_normalize_range_dict():
    assert _normalize_range({"min": 0.1, "max": 0.5}) == (0.1, 0.5)


def test_normalize_range_invalid():
    with pytest.raises(InvalidParameterSpecError):
        _normalize_range([0.1])
    with pytest.raises(InvalidParameterSpecError):
        _normalize_range("bad")


def test_choose_default_precedence():
    # Explicit > live > midpoint
    assert _choose_default(0.42, 0.1, 0.0, 1.0, "float") == pytest.approx(0.42)
    assert _choose_default(None, 0.1, 0.0, 1.0, "float") == pytest.approx(0.1)
    assert _choose_default(None, None, 0.0, 1.0, "float") == pytest.approx(0.5)


def test_choose_default_int_coerces():
    # Explicit 2.7 → 3 (round)
    assert _choose_default(2.7, None, 1, 5, "int") == 3
    # Midpoint of 1..5 = 3
    assert _choose_default(None, None, 1, 5, "int") == 3


# ==========================================================================
# Integration — loaded space is usable by proposer
# ==========================================================================


def test_loaded_space_usable_by_random_proposer():
    """End-to-end: load YAML → feed SearchSpace to RandomProposer →
    get a valid config."""
    import random
    from src.calibration.proposer import RandomProposer

    loaded = load_default_search_space()
    proposer = RandomProposer(loaded.space, seed=42)
    config = proposer.propose([])
    # Every spec's default key is in the random config and within bounds
    for spec in loaded.space.specs:
        assert spec.name in config
        if spec.kind == "float":
            assert spec.lo <= config[spec.name] <= spec.hi
        elif spec.kind == "int":
            assert spec.lo <= config[spec.name] <= spec.hi


def test_loaded_space_validates_default_config():
    loaded = load_default_search_space()
    default_cfg = loaded.space.default_config()
    ok, errs = loaded.space.validate_config(default_cfg)
    assert ok, f"default config is invalid: {errs}"


# ==========================================================================
# Profile validation (Phase 4.5 corrective pass)
# ==========================================================================


def test_unknown_profile_delta_section_raises(tmp_path):
    """A `<profile>_deltas` section whose resolved profile name is not
    in the live PROFILE_DELTAS must raise InvalidParameterSpecError at
    load time, not silently produce dotted keys that fail downstream."""
    fake_yaml = tmp_path / "typo_profile.yaml"
    fake_yaml.write_text(
        """
made_up_profile_deltas:
  cognitive:
    att_bandwidth:
      range: [-3, -1]
      default: -2
""",
        encoding="utf-8",
    )
    with pytest.raises(InvalidParameterSpecError) as exc_info:
        load_search_space(fake_yaml)
    msg = str(exc_info.value)
    assert "made_up_profile_deltas" in msg
    assert "made_up_profile" in msg
    assert "Known profiles" in msg


def test_typo_profile_delta_section_raises(tmp_path):
    """Common typo: `adhd_inatentive_deltas` (missing 't')."""
    fake_yaml = tmp_path / "typo_adhd.yaml"
    fake_yaml.write_text(
        """
adhd_inatentive_deltas:
  cognitive:
    att_bandwidth:
      range: [-3, -1]
""",
        encoding="utf-8",
    )
    with pytest.raises(InvalidParameterSpecError, match="adhd_inatentive"):
        load_search_space(fake_yaml)


def test_known_alias_still_resolves(tmp_path):
    """`adhd_hyperactive_deltas` must still resolve to the canonical
    `adhd_hyperactive_impulsive` profile via the alias map."""
    fake_yaml = tmp_path / "alias.yaml"
    fake_yaml.write_text(
        """
adhd_hyperactive_deltas:
  cognitive:
    impulse_override:
      range: [0.25, 0.40]
      default: 0.325
""",
        encoding="utf-8",
    )
    loaded = load_search_space(fake_yaml)
    names = loaded.space.names()
    assert "adhd_hyperactive_impulsive.cognitive.impulse_override" in names


def test_all_canonical_profiles_load_without_validation_error(tmp_path):
    """Every profile in PROFILE_DELTAS should be accepted by the loader
    when used as a section name."""
    from src.simulation import cognitive_agent as ca

    # Build a minimal YAML with one delta entry per known profile
    lines = []
    for profile in sorted(ca.PROFILE_DELTAS.keys()):
        # Respect alias direction: the YAML typically uses the canonical
        # name directly (e.g. adhd_inattentive_deltas) so we just append
        # _deltas to each.
        lines.append(f"{profile}_deltas:")
        lines.append("  emotional:")
        lines.append("    frustration:")
        lines.append("      range: [-0.1, 0.3]")
    fake_yaml = tmp_path / "all_profiles.yaml"
    fake_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Should load cleanly — every profile exists
    loaded = load_search_space(fake_yaml)
    assert len(loaded.space) == len(ca.PROFILE_DELTAS)


def test_real_harness_profiles_validate():
    """The real harness YAML's profile sections must validate clean."""
    # Already covered by test_load_default_returns_non_empty_space, but this
    # test is explicit about the validation gate.
    loaded = load_default_search_space()
    # If validation is working, all profile delta entries in loaded.space
    # must correspond to dotted keys whose profile is in PROFILE_DELTAS.
    from src.simulation import cognitive_agent as ca
    for spec in loaded.space.specs:
        parts = spec.name.split(".")
        if len(parts) == 3:
            # profile_delta path
            profile = parts[0]
            assert profile in ca.PROFILE_DELTAS, (
                f"loaded spec {spec.name!r} references unknown profile {profile!r}"
            )

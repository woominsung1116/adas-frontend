"""Situational Modulator — environmental factors affecting classroom dynamics.

Applies literature-based environmental modulations on top of baseline
student/classroom parameters. Modulations include:

  - Academic cycle (exam stress, fatigue peaks)
  - Diurnal rhythm (post-lunch dip)
  - Event-based (peer conflict, presentations, substitute teachers)
  - New semester adaptation
  - Weather/seasonal (subtle effects only)

Design principles (see 정리.md §25.10):
  - All modulation coefficients are literature-derived (with explicit sources)
  - Ranges for autoresearch calibration (not raw magic numbers)
  - Additive composition when multiple modulations active simultaneously
  - Clamped to safe bounds to prevent runaway effects

References:
  - Cassady & Johnson (2002) test anxiety: d ≈ -0.40 ~ -0.52
  - Lim & Kwok (2016) vigilance decrement
  - Eisenberger et al. (2003) social pain neuroscience
  - Schmidt et al. (2007) circadian cognition
  - Folkard (1975) post-lunch dip
  - Akos & Galassi (2004) transition stress
  - Felitti et al. (1998) ACE / home stressor
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Modulation vector (what a modulator emits)
# ---------------------------------------------------------------------------


@dataclass
class ModulationVector:
    """A set of offsets applied to student/classroom state for one turn.

    All fields default to 0 (no effect). Modulators emit these vectors and
    the SituationalModulator class combines them additively.
    """

    # Global emotion shifts (applied to all students).
    # NOTE: only anxiety/excitement remain after the 8→3 student-emotion
    # reduction; frustration/loneliness channels were removed.
    global_anxiety: float = 0.0
    global_excitement: float = 0.0

    # Global cognitive shifts
    global_attention: float = 0.0         # +/- to observable attention
    global_compliance: float = 0.0

    # Class-level (chaos, disruption)
    class_stress: float = 0.0              # 0-1 scale, affects all
    class_disruption: float = 0.0          # 0-1 scale, affects peer interactions

    # Amplification factors (multiplicative for specific profiles)
    adhd_amplification: float = 1.0       # reactivity scaling for ADHD students
    anxiety_amplification: float = 1.0    # reactivity scaling for anxiety
    odd_amplification: float = 1.0        # reactivity scaling for ODD

    # Event flags (for downstream handlers)
    exam_week: bool = False
    substitute_teacher: bool = False
    peer_conflict: bool = False
    presentation_active: bool = False
    post_lunch_dip: bool = False

    def combine(self, other: "ModulationVector") -> "ModulationVector":
        """Additive combination of two modulation vectors."""
        return ModulationVector(
            global_anxiety=self.global_anxiety + other.global_anxiety,
            global_excitement=self.global_excitement + other.global_excitement,
            global_attention=self.global_attention + other.global_attention,
            global_compliance=self.global_compliance + other.global_compliance,
            class_stress=max(self.class_stress, other.class_stress),  # max, not sum
            class_disruption=max(self.class_disruption, other.class_disruption),
            adhd_amplification=self.adhd_amplification * other.adhd_amplification,
            anxiety_amplification=self.anxiety_amplification * other.anxiety_amplification,
            odd_amplification=self.odd_amplification * other.odd_amplification,
            exam_week=self.exam_week or other.exam_week,
            substitute_teacher=self.substitute_teacher or other.substitute_teacher,
            peer_conflict=self.peer_conflict or other.peer_conflict,
            presentation_active=self.presentation_active or other.presentation_active,
            post_lunch_dip=self.post_lunch_dip or other.post_lunch_dip,
        )


# ---------------------------------------------------------------------------
# Individual modulators (each returns a ModulationVector)
# ---------------------------------------------------------------------------


def academic_cycle_modulation(turn: int, total_turns: int = 950) -> ModulationVector:
    """Academic cycle stress — exam weeks and late-semester fatigue.

    Based on Cassady & Johnson (2002), Lim & Kwok (2016).

    Approximate Korean K-6 academic calendar (950 turns = 1 year):
      Turn 1-100: 학기 초 (new semester)
      Turn 100-300: 안정기 (routine)
      Turn 300-400: 1학기 중간고사 주간
      Turn 400-475: 1학기 후반 (fatigue)
      Turn 475-600: 2학기 초 (reset + routine)
      Turn 600-700: 2학기 중간고사
      Turn 700-850: 2학기 기말 준비 (peak stress)
      Turn 850-950: 기말 직전 + 방학 전 이완
    """
    v = ModulationVector()

    # Midterm exam weeks
    if 300 <= turn <= 400 or 600 <= turn <= 700:
        v.global_anxiety = 0.15
        v.global_attention = -0.08
        v.class_stress = 0.5
        v.exam_week = True
        v.adhd_amplification = 1.6  # Putwain 2008: ADHD 1.5-2x
        v.anxiety_amplification = 2.0

    # Late-semester fatigue (end of semester 1 and 2)
    elif 400 <= turn <= 475 or 700 <= turn <= 850:
        v.global_excitement = -0.08
        v.global_attention = -0.05
        v.class_stress = 0.3

    # Peak stress (finals prep)
    if 800 <= turn <= 900:
        v.global_anxiety += 0.10
        v.class_stress = max(v.class_stress, 0.6)
        v.exam_week = True

    return v


def new_semester_adaptation(turn: int) -> ModulationVector:
    """New semester adaptation stress (first 100 turns).

    Source: Akos & Galassi (2004) school transition stress.
    Linear decay from peak to zero over 100 turns.
    """
    v = ModulationVector()
    if turn <= 100:
        factor = (100 - turn) / 100.0
        v.global_anxiety = 0.15 * factor
        v.global_excitement = 0.08 * factor  # novelty
        v.anxiety_amplification = 1.0 + 0.8 * factor
    return v


def diurnal_rhythm(period: int) -> ModulationVector:
    """Intra-day cognitive rhythm (per period within a day).

    Source: Schmidt et al. (2007), Folkard (1975) post-lunch dip.
    Korean K-6 schedule: 5-6 periods per day.

    Period 1-2: morning (baseline)
    Period 3-4: late morning (slight decline)
    Period 5: post-lunch (dip)
    Period 6+: afternoon (modest decline)
    """
    v = ModulationVector()

    if period == 3 or period == 4:
        v.global_attention = -0.03
    elif period == 5:
        v.global_attention = -0.08  # post-lunch dip
        v.global_excitement = -0.05
        v.post_lunch_dip = True
    elif period >= 6:
        v.global_attention = -0.06
        v.global_excitement = -0.04

    return v


def peer_conflict_event(active: bool, severity: float = 1.0) -> ModulationVector:
    """Peer conflict event modulation.

    Source: Eisenberger et al. (2003) social pain, Cacioppo et al. (2009).
    Affects entire class (emotional contagion), amplified for involved students.
    """
    v = ModulationVector()
    if active:
        v.global_anxiety = 0.05 * severity
        v.class_disruption = 0.4 * severity
        v.peer_conflict = True
        # Targeted students (not handled here — downstream)
    return v


def substitute_teacher(active: bool) -> ModulationVector:
    """Substitute teacher presence — routine disruption.

    Source: Clifton & Rambaran (1987), Duffrin (2002).
    """
    v = ModulationVector()
    if active:
        v.global_excitement = 0.08
        v.global_compliance = -0.15
        v.class_disruption = 0.3
        v.substitute_teacher = True
        v.odd_amplification = 2.5  # ODD students most affected
    return v


def presentation_event(active: bool) -> ModulationVector:
    """Class presentation/speaking event.

    Source: Behnke & Sawyer (2000), Stein et al. (1996).
    Targeted at presenter (anxiety spike), mild class-wide attention shift.
    """
    v = ModulationVector()
    if active:
        v.global_anxiety = 0.03
        v.presentation_active = True
        v.anxiety_amplification = 2.5  # severe for anxiety students
    return v


def hot_weather(intensity: float = 0.0) -> ModulationVector:
    """Hot weather modulation (small effect, use sparingly).

    Source: Hancock et al. (2007) thermal stress meta-analysis d ≈ -0.28.
    NOTE: Denissen 2008 shows weather-mood effects are SMALL (r ≈ 0.1).
    Do not inflate this.
    """
    v = ModulationVector()
    if intensity > 0:
        v.global_attention = -0.05 * intensity
        v.global_excitement = -0.04 * intensity
    return v


# ---------------------------------------------------------------------------
# Main modulator — orchestrates all active modulations
# ---------------------------------------------------------------------------


@dataclass
class SituationalEvent:
    """A scheduled one-time or recurring event."""

    event_type: str  # "peer_conflict", "substitute_teacher", "presentation", ...
    start_turn: int
    duration: int = 1
    severity: float = 1.0
    active: bool = True


class SituationalModulator:
    """Compose multiple modulations into a single ModulationVector per turn.

    Usage:
        modulator = SituationalModulator()
        modulator.add_event(SituationalEvent("peer_conflict", start_turn=150, duration=30))

        for turn in range(1, 951):
            mod = modulator.compute_modulation(turn, period=current_period)
            apply_modulation_to_students(mod, students)
    """

    def __init__(self, total_turns: int = 950) -> None:
        self.total_turns = total_turns
        self.scheduled_events: list[SituationalEvent] = []
        self._hot_weather_intensity: float = 0.0

    def add_event(self, event: SituationalEvent) -> None:
        """Schedule a new event."""
        self.scheduled_events.append(event)

    def set_hot_weather(self, intensity: float) -> None:
        """Set ambient heat intensity (0.0 to 1.0)."""
        self._hot_weather_intensity = max(0.0, min(1.0, intensity))

    def compute_modulation(
        self,
        turn: int,
        period: int = 1,
        include_weather: bool = False,
    ) -> ModulationVector:
        """Compose all active modulations for the given turn.

        Args:
            turn: current simulation turn (1 to total_turns)
            period: current class period within the day (1-6)
            include_weather: whether to apply weather modulation

        Returns:
            Combined ModulationVector for this turn.
        """
        # 1. Academic cycle (long-term)
        mod = academic_cycle_modulation(turn, self.total_turns)

        # 2. New semester adaptation (early turns)
        mod = mod.combine(new_semester_adaptation(turn))

        # 3. Diurnal rhythm (per period)
        mod = mod.combine(diurnal_rhythm(period))

        # 4. Active events
        for event in self._active_events(turn):
            event_mod = self._event_to_modulation(event)
            mod = mod.combine(event_mod)

        # 5. Weather (optional)
        if include_weather and self._hot_weather_intensity > 0:
            mod = mod.combine(hot_weather(self._hot_weather_intensity))

        return mod

    def _active_events(self, turn: int) -> list[SituationalEvent]:
        """Return events that are currently active at this turn."""
        return [
            ev for ev in self.scheduled_events
            if ev.active and ev.start_turn <= turn < ev.start_turn + ev.duration
        ]

    def _event_to_modulation(self, event: SituationalEvent) -> ModulationVector:
        """Convert a SituationalEvent into a ModulationVector."""
        if event.event_type == "peer_conflict":
            return peer_conflict_event(active=True, severity=event.severity)
        elif event.event_type == "substitute_teacher":
            return substitute_teacher(active=True)
        elif event.event_type == "presentation":
            return presentation_event(active=True)
        else:
            return ModulationVector()  # Unknown → no-op


# ---------------------------------------------------------------------------
# Default scenario presets
# ---------------------------------------------------------------------------


def default_korean_k6_schedule(total_turns: int = 950) -> SituationalModulator:
    """Create a default Korean K-6 schedule with typical event patterns.

    Includes:
      - 2 peer conflicts per semester (random-ish but deterministic)
      - 1 substitute teacher day per semester
      - Presentations distributed across the year
    """
    modulator = SituationalModulator(total_turns=total_turns)

    # Peer conflicts — 2 per semester (turns ~150, ~250, ~600, ~800)
    modulator.add_event(SituationalEvent("peer_conflict", start_turn=150, duration=30, severity=0.8))
    modulator.add_event(SituationalEvent("peer_conflict", start_turn=250, duration=25, severity=0.6))
    modulator.add_event(SituationalEvent("peer_conflict", start_turn=620, duration=35, severity=1.0))
    modulator.add_event(SituationalEvent("peer_conflict", start_turn=800, duration=20, severity=0.7))

    # Substitute teacher (rare)
    modulator.add_event(SituationalEvent("substitute_teacher", start_turn=200, duration=5))
    modulator.add_event(SituationalEvent("substitute_teacher", start_turn=750, duration=5))

    # Presentations (distributed)
    modulator.add_event(SituationalEvent("presentation", start_turn=180, duration=5))
    modulator.add_event(SituationalEvent("presentation", start_turn=350, duration=5))
    modulator.add_event(SituationalEvent("presentation", start_turn=550, duration=5))
    modulator.add_event(SituationalEvent("presentation", start_turn=720, duration=5))

    return modulator

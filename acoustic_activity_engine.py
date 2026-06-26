import librosa
import numpy as np
import json
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment


# =========================
# DATACLASSES
# =========================

@dataclass
class AcousticObservation:
    timestamp: float
    energy: float
    signature_vector: np.ndarray
    log_f0: float  # log-normalised pitch; -1.0 if unvoiced


@dataclass
class AcousticEntity:
    entity_id: int
    signature_vector: np.ndarray
    log_f0: float
    last_activity_timestamp: float
    first_activity_timestamp: float
    active_frame_count: int = 0


@dataclass
class ActivityEvent:
    timestamp: float
    entity_id: int
    event_type: str

    def __str__(self):
        return f"{self.timestamp:.2f}s : Entity_{self.entity_id} {self.event_type}"


# =========================
# HELPERS
# =========================

def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def _estimate_f0(chunk: np.ndarray, sr: int,
                 fmin: int = 60, fmax: int = 400,
                 confidence_threshold: float = 0.3) -> float:
    """
    Autocorrelation-based fundamental frequency estimator.
    Returns Hz if a confident voiced pitch is found, else 0.0.

    Works on raw speech without any ML model — purely first-principles:
    the autocorrelation of a periodic signal peaks at multiples of its period.
    """
    n = len(chunk)
    if n < sr // fmin:
        return 0.0

    windowed = chunk * np.hanning(n)
    corr = np.correlate(windowed, windowed, mode='full')[n - 1:]

    lag_min = int(sr / fmax)
    lag_max = min(int(sr / fmin), len(corr) - 1)
    if lag_max <= lag_min:
        return 0.0

    peak_lag = int(np.argmax(corr[lag_min:lag_max])) + lag_min
    confidence = corr[peak_lag] / (corr[0] + 1e-10)
    if confidence < confidence_threshold:
        return 0.0

    return sr / peak_lag if peak_lag > 0 else 0.0


def _log_f0_normalised(f0_hz: float, fmin: float = 60.0, fmax: float = 400.0) -> float:
    """
    Map raw Hz onto a [0, 1] log scale spanning the human voice range.
    Returns -1.0 for unvoiced frames so callers can detect absence of pitch.
    """
    if f0_hz <= 0:
        return -1.0
    return np.log(f0_hz / fmin) / np.log(fmax / fmin)


def _combined_similarity(mfcc_a: np.ndarray, lf0_a: float,
                          mfcc_b: np.ndarray, lf0_b: float,
                          mfcc_alpha: float) -> float:
    """
    Weighted combination of MFCC cosine similarity and pitch proximity.

    MFCC alone cannot separate speakers with similar vocal tract shapes —
    pitch is the dominant cue for male/female separation.  We keep MFCC
    for fine-grained identity within a gender class and let pitch carry
    the cross-gender discrimination.

    pitch_sim = 1 - |lf0_a - lf0_b|  (both on [0,1] log scale, so max
    difference is 1.0, meaning the term spans [0, 1] naturally)

    If either frame is unvoiced (lf0 == -1) we fall back to MFCC only
    so unvoiced frames don't wrongly discriminate.
    """
    mfcc_sim = float(np.dot(mfcc_a, mfcc_b))

    if lf0_a >= 0.0 and lf0_b >= 0.0:
        pitch_sim = 1.0 - abs(lf0_a - lf0_b)
        return mfcc_alpha * mfcc_sim + (1.0 - mfcc_alpha) * pitch_sim
    else:
        return mfcc_sim


# =========================
# EXTRACTOR
# =========================

class Extractor:
    def __init__(self, global_settings: dict, acoustic_profile: dict):

        self.window_duration        = acoustic_profile["window_duration"]
        self.minimum_energy         = acoustic_profile["minimum_energy"]
        self.n_mfcc                 = acoustic_profile["n_mfcc"]
        self.use_delta_mfcc         = acoustic_profile.get("use_delta_mfcc", False)
        # How many hops to average for the spectral signature.
        # A single 0.6 s MFCC window captures phoneme content, not speaker
        # identity, producing consecutive-window similarities as low as -0.67.
        # Averaging over 2 hops (1.2 s) raises the floor to ~0.59.
        self.context_window_multiplier = acoustic_profile.get("context_window_multiplier", 2)
        # F0 sub-frame length (seconds) for pitch estimation.
        self.f0_frame_duration      = acoustic_profile.get("f0_frame_duration", 0.3)

        self.sample_rate = 22050

    def process_audio(self, audio_path: str):
        audio_signal, sample_rate = librosa.load(audio_path, sr=self.sample_rate)
        hop_size     = int(self.window_duration  * sample_rate)
        context_size = hop_size * self.context_window_multiplier
        f0_frame     = int(self.f0_frame_duration * sample_rate)

        observation_batches = []

        for start in range(0, len(audio_signal), hop_size):

            hop_window = audio_signal[start:start + hop_size]
            if len(hop_window) == 0:
                continue

            energy = float(np.mean(hop_window ** 2))
            if energy < self.minimum_energy:
                continue

            ctx_end        = min(len(audio_signal), start + context_size)
            context_window = audio_signal[start:ctx_end]

            # ── MFCC signature ──────────────────────────────────────────────
            mfcc = librosa.feature.mfcc(
                y=context_window,
                sr=sample_rate,
                n_mfcc=self.n_mfcc
            )
            mfcc_vector = np.mean(mfcc, axis=1)

            if self.use_delta_mfcc:
                delta       = librosa.feature.delta(mfcc)
                delta_vector = np.mean(np.abs(delta), axis=1)
                mfcc_vector  = np.concatenate((mfcc_vector, delta_vector))

            mfcc_vector = _normalize(mfcc_vector)

            # ── Pitch signature ─────────────────────────────────────────────
            # Estimate F0 on each short sub-frame within the context window
            # and average voiced frames only.  Using sub-frames rather than
            # the full context gives better time resolution on pitch.
            f0_values = []
            for s in range(0, len(context_window) - f0_frame + 1, f0_frame):
                f0 = _estimate_f0(context_window[s:s + f0_frame], sample_rate)
                if f0 > 0:
                    f0_values.append(f0)

            mean_f0 = float(np.mean(f0_values)) if f0_values else 0.0
            log_f0  = _log_f0_normalised(mean_f0)

            observation_batches.append(AcousticObservation(
                timestamp        = start / sample_rate,
                energy           = energy,
                signature_vector = mfcc_vector,
                log_f0           = log_f0,
            ))

        return observation_batches


# =========================
# ENTITY TRACKER
# =========================

class EntityTracker:
    def __init__(self, global_settings: dict, acoustic_profile: dict):

        self.memory_limit            = global_settings.get("memory_limit", 10)
        self.smoothing_alpha         = global_settings.get("signature_smoothing_alpha", 0.7)
        self.min_confirmation_frames = global_settings.get("min_confirmation_frames", 2)
        # Weight given to MFCC vs pitch in combined similarity.
        self.mfcc_alpha              = global_settings.get("mfcc_alpha", 0.3)

        self.similarity_threshold    = acoustic_profile["similarity_threshold"]

        self.next_entity_id = 0
        self.entities       = {}
        self.disappeared    = {}
        self._pending: dict[int, dict] = {}
        self._next_pending_id = 0

    def _sim(self, entity_or_pending: dict, obs: AcousticObservation) -> float:
        return _combined_similarity(
            entity_or_pending["vector"], entity_or_pending["log_f0"],
            obs.signature_vector,        obs.log_f0,
            self.mfcc_alpha,
        )

    def _ema_update(self, old_vec, old_lf0, new_vec, new_lf0):
        a = self.smoothing_alpha
        updated_vec = _normalize(a * old_vec + (1 - a) * new_vec)
        # Only blend pitch if both frames are voiced
        if old_lf0 >= 0 and new_lf0 >= 0:
            updated_lf0 = a * old_lf0 + (1 - a) * new_lf0
        else:
            updated_lf0 = old_lf0
        return updated_vec, updated_lf0

    def _register(self, vector, log_f0, timestamp):
        eid = self.next_entity_id
        self.entities[eid] = AcousticEntity(
            entity_id               = eid,
            signature_vector        = _normalize(vector.copy()),
            log_f0                  = log_f0,
            last_activity_timestamp = timestamp,
            first_activity_timestamp= timestamp,
            active_frame_count      = 1,
        )
        self.disappeared[eid] = 0
        self.next_entity_id  += 1

    def _deregister(self, entity_id):
        del self.entities[entity_id]
        del self.disappeared[entity_id]

    def update(self, observations: list[AcousticObservation]):

        # ── No observations ──────────────────────────────────────────────────
        if not observations:
            for eid in list(self.disappeared):
                self.disappeared[eid] += 1
                if self.disappeared[eid] >= self.memory_limit:
                    self._deregister(eid)
            for pid in list(self._pending):
                self._pending[pid]["missed"] += 1
                if self._pending[pid]["missed"] >= self.memory_limit:
                    del self._pending[pid]
            return self.entities

        entity_ids = list(self.entities.keys())

        # ── Match observations → confirmed entities ───────────────────────
        used_entities = set()
        used_obs      = set()

        if entity_ids:
            sim_matrix = np.array([
                [self._sim({"vector": self.entities[eid].signature_vector,
                            "log_f0": self.entities[eid].log_f0}, obs)
                 for obs in observations]
                for eid in entity_ids
            ])

            cost_matrix = 1.0 - sim_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_ind, col_ind):
                s = sim_matrix[r, c]
                if s < 0.2:
                    continue
                if s <= self.similarity_threshold:
                    continue

                eid = entity_ids[r]
                ent = self.entities[eid]

                if ent.active_frame_count < 2 and s < 0.90:
                    continue

                new_vec, new_lf0 = self._ema_update(
                    ent.signature_vector, ent.log_f0,
                    observations[c].signature_vector, observations[c].log_f0,
                )
                ent.signature_vector        = new_vec
                ent.log_f0                  = new_lf0
                ent.last_activity_timestamp = observations[c].timestamp
                ent.active_frame_count     += 1
                self.disappeared[eid]       = 0

                used_entities.add(r)
                used_obs.add(c)

        # ── Age unmatched confirmed entities ─────────────────────────────
        for idx, eid in enumerate(entity_ids):
            if idx in used_entities:
                continue
            self.disappeared[eid] += 1
            if self.disappeared[eid] >= self.memory_limit:
                self._deregister(eid)

        # ── Unmatched observations → pending ────────────────────────────
        unmatched = [observations[i] for i in range(len(observations)) if i not in used_obs]

        pending_ids      = list(self._pending.keys())
        existing_pending = set(pending_ids)

        used_pending   = set()
        used_unmatched = set()

        if pending_ids and unmatched:
            p_sim_matrix = np.array([
                [self._sim(self._pending[pid], obs) for obs in unmatched]
                for pid in pending_ids
            ])
            cost_matrix  = 1.0 - p_sim_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_ind, col_ind):
                s = p_sim_matrix[r, c]
                if s < 0.2 or s <= self.similarity_threshold:
                    continue

                pid = pending_ids[r]
                obs = unmatched[c]

                new_vec, new_lf0 = self._ema_update(
                    self._pending[pid]["vector"], self._pending[pid]["log_f0"],
                    obs.signature_vector, obs.log_f0,
                )
                self._pending[pid]["vector"] = new_vec
                self._pending[pid]["log_f0"] = new_lf0
                self._pending[pid]["count"] += 1
                self._pending[pid]["missed"] = 0

                if self._pending[pid]["count"] >= self.min_confirmation_frames:
                    self._register(
                        self._pending[pid]["vector"],
                        self._pending[pid]["log_f0"],
                        self._pending[pid]["first_ts"],
                    )
                    self.entities[self.next_entity_id - 1].last_activity_timestamp = obs.timestamp
                    del self._pending[pid]

                used_pending.add(r)
                used_unmatched.add(c)

        # New pending candidates for still-unmatched observations
        for i, obs in enumerate(unmatched):
            if i in used_unmatched:
                continue
            self._pending[self._next_pending_id] = {
                "vector":   _normalize(obs.signature_vector.copy()),
                "log_f0":   obs.log_f0,
                "first_ts": obs.timestamp,
                "count":    1,
                "missed":   0,
            }
            self._next_pending_id += 1

        # ── Age stale pending ────────────────────────────────────────────
        for pid in list(self._pending):
            if pid not in existing_pending:
                continue
            if pid not in self._pending:
                continue
            self._pending[pid]["missed"] += 1
            if self._pending[pid]["missed"] >= self.memory_limit:
                del self._pending[pid]

        return self.entities


# =========================
# ACTIVITY RESOLVER
# =========================

class ActivityResolver:
    def __init__(self, acoustic_profile: dict, global_settings: dict):
        self.memory                   = {}
        self.minimum_activity_duration = acoustic_profile["minimum_activity_duration"]
        self.window_duration           = acoustic_profile["window_duration"]

    def update(self, entities: dict):
        events  = []
        current = set(entities.keys())
        prev    = set(self.memory.keys())

        for eid, ent in entities.items():
            if eid not in self.memory:
                duration = ent.last_activity_timestamp - ent.first_activity_timestamp
                if duration >= self.minimum_activity_duration:
                    events.append(ActivityEvent(ent.first_activity_timestamp, eid, "STARTED"))
                    self.memory[eid] = ent.last_activity_timestamp
            else:
                self.memory[eid] = ent.last_activity_timestamp

        for eid in prev - current:
            stopped_at = self.memory[eid] + self.window_duration
            events.append(ActivityEvent(stopped_at, eid, "STOPPED"))
            del self.memory[eid]

        return events


# =========================
# ENGINE
# =========================

def run_engine(profile_key: str, audio_source: str):

    with open("engine_settings.json") as f:
        global_settings = json.load(f)["global_settings"]

    with open("acoustic_profiles.json") as f:
        acoustic_profile = json.load(f)[profile_key]

    extractor = Extractor(global_settings, acoustic_profile)
    tracker   = EntityTracker(global_settings, acoustic_profile)
    resolver  = ActivityResolver(acoustic_profile, global_settings)

    print(f"\n[INFO] Profile: {profile_key.upper()}")
    print(f"[INFO] Audio: {audio_source}\n")

    batches = extractor.process_audio(audio_source)
    events  = []

    for obs in batches:
        entities = tracker.update([obs])
        for e in resolver.update(entities):
            print(e)
            events.append(e)

    final_entities = tracker.update([])
    for e in resolver.update(final_entities):
        print(e)
        events.append(e)

    while resolver.memory:
        final_entities = tracker.update([])
        for e in resolver.update(final_entities):
            print(e)
            events.append(e)

    print("\n==============================")
    print("ACTIVITY SUMMARY")
    print("==============================")
    print("Profile:", profile_key)
    print("Total Events:", len(events))
    print("==============================\n")


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    print("=" * 50)
    print("ACOUSTIC ACTIVITY ENGINE")
    print("=" * 50)

    with open("acoustic_profiles.json") as f:
        profiles = list(json.load(f).keys())

    print("Available:", profiles)

    profile = input("Profile: ").strip()
    if not profile or profile not in profiles:
        profile = profiles[0]

    audio = input("Audio file: ").strip() or "sample_audio.wav"

    run_engine(profile, audio)

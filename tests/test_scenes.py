"""Scene catalogue: what a scene captures, and how the store keeps it on disk."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmon3 import config, prompt                              # noqa: E402
from harmon3.graph_builder import BuildState                    # noqa: E402
from harmon3.refs import AUDIO, IMAGE, VIDEO, RefRow, RefSet    # noqa: E402
from harmon3.scenes import (                                    # noqa: E402
    MAX_DESCRIPTION_LENGTH,
    PROJECTS_FILENAME,
    UNGROUPED,
    Scene,
    SceneStore,
    SceneWriteError,
    clean_description,
    signature_of,
    slugify,
)


@pytest.fixture
def store(tmp_path):
    return SceneStore(tmp_path / "scenes")


def _state(**kw):
    state = BuildState(
        prompt_sections=prompt.from_legacy(kw.get("prompt", "Hero shot of <Picture 1>.")),
        duration_seconds=kw.get("duration", 6.0),
        seed=kw.get("seed", 4242),
        refs=kw.get("refs", RefSet(
            images=[RefRow(kind=IMAGE, comfy_name="hero.png")],
            videos=[RefRow(kind=VIDEO, comfy_name="pan.mp4", use_soundtrack=True)],
            audios=[RefRow(kind=AUDIO, comfy_name="roar.wav")],
        )),
    )
    state.aspect_ratio = kw.get("aspect", state.aspect_ratio)
    state.megapixels = kw.get("megapixels", state.megapixels)
    return state


# ---------------------------------------------------------------------------------
# What a scene captures
# ---------------------------------------------------------------------------------

def test_scene_captures_the_four_defining_fields():
    scene = Scene.from_state("Opening", _state(), randomize_seed=False)
    assert scene.name == "Opening"
    assert scene.prompt_sections["detailed_description"] == "Hero shot of <Picture 1>."
    assert scene.duration_seconds == 6.0
    assert scene.seed == 4242
    assert scene.randomize_seed is False
    assert [r["kind"] for r in scene.refs] == ["image", "video", "audio"]


def test_scene_round_trips_through_a_state():
    original = _state()
    scene = Scene.from_state("Opening", original, randomize_seed=True)

    restored = BuildState()
    scene.apply_to_state(restored)

    assert restored.prompt_text == original.prompt_text
    assert restored.duration_seconds == original.duration_seconds
    assert restored.seed == original.seed
    assert [r.comfy_name for r in restored.refs.all_rows()] == \
           [r.comfy_name for r in original.refs.all_rows()]
    assert restored.refs.videos[0].use_soundtrack is True


def test_loading_a_scene_leaves_render_settings_alone():
    """Resolution and sampling are render settings, so a scene can run draft then final."""
    scene = Scene.from_state("Opening", _state(), randomize_seed=True)

    target = BuildState()
    target.aspect_ratio = "21:9 (Ultrawide)"
    target.megapixels = 2.0
    target.steps = 40
    target.ref_image_size = "max"
    scene.apply_to_state(target)

    assert target.aspect_ratio == "21:9 (Ultrawide)"
    assert target.megapixels == 2.0
    assert target.steps == 40
    assert target.ref_image_size == "max"


def test_soundtrack_choice_is_part_of_the_scene():
    """It changes the audio tag ordinals, so it has to travel with the scene."""
    state = _state()
    state.refs.videos[0].use_soundtrack = False
    scene = Scene.from_state("Silent", state, randomize_seed=True)

    restored = BuildState()
    scene.apply_to_state(restored)
    assert restored.refs.videos[0].use_soundtrack is False


def test_local_reference_paths_survive_but_upload_names_do_not():
    state = _state(refs=RefSet(images=[RefRow(kind=IMAGE, local_path="C:/refs/a.png")]))
    state.refs.images[0].comfy_name = "harmon3/a_deadbeef.png"
    scene = Scene.from_state("Local", state, randomize_seed=True)

    entry = scene.refs[0]
    assert entry["local_path"] == "C:/refs/a.png"
    assert "comfy_name" not in entry          # recomputed from the file's contents


# ---------------------------------------------------------------------------------
# Modified detection
# ---------------------------------------------------------------------------------

def test_a_freshly_saved_scene_is_not_modified():
    state = _state()
    scene = Scene.from_state("Opening", state, randomize_seed=True)
    assert scene.matches_state(state, randomize_seed=True)


def test_editing_the_prompt_marks_it_modified():
    state = _state()
    scene = Scene.from_state("Opening", state, randomize_seed=True)
    state.prompt_sections["summary"] = "Now with more drama."
    assert not scene.matches_state(state, randomize_seed=True)


def test_editing_the_duration_marks_it_modified():
    state = _state()
    scene = Scene.from_state("Opening", state, randomize_seed=True)
    state.duration_seconds = 9.0
    assert not scene.matches_state(state, randomize_seed=True)


def test_editing_the_seed_marks_a_fixed_seed_scene_modified():
    state = _state()
    scene = Scene.from_state("Opening", state, randomize_seed=False)
    state.seed = 999
    assert not scene.matches_state(state, randomize_seed=False)


def test_a_rolled_seed_does_not_mark_a_randomized_scene_modified():
    """Every queue rolls a new seed, so for a randomized scene the number is not the setting.

    Without this, running a scene once would immediately show it as edited.
    """
    state = _state()
    scene = Scene.from_state("Opening", state, randomize_seed=True)
    state.seed = 987654321
    assert scene.matches_state(state, randomize_seed=True)


def test_toggling_randomize_marks_it_modified():
    state = _state()
    scene = Scene.from_state("Opening", state, randomize_seed=True)
    assert not scene.matches_state(state, randomize_seed=False)


def test_changing_render_settings_does_not_mark_it_modified():
    """A scene owns none of these, so changing one must not look like an edit."""
    state = _state()
    scene = Scene.from_state("Opening", state, randomize_seed=True)
    state.aspect_ratio = "1:1 (Square)"
    state.megapixels = 1.6
    state.steps = 40
    state.ref_image_size = "max"
    assert scene.matches_state(state, randomize_seed=True)


def test_probe_results_do_not_mark_it_modified():
    """Probes fill in fps and frame counts; that is discovery, not an edit."""
    state = _state(refs=RefSet(videos=[RefRow(kind=VIDEO, comfy_name="pan.mp4")]))
    scene = Scene.from_state("Opening", state, randomize_seed=True)
    row = state.refs.videos[0]
    row.fps, row.frame_count, row.has_audio, row.width = 24.0, 300, True, 1920
    assert scene.matches_state(state, randomize_seed=True)


def test_signature_ignores_float_noise():
    assert signature_of("p", 5.0, 1, True, []) == signature_of("p", 5.00000001, 1, True, [])


# ---------------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------------

def test_save_and_reload_the_catalogue(store):
    store.save(Scene.from_state("Opening shot", _state(), randomize_seed=False))
    store.save(Scene.from_state("Closing shot", _state(prompt="The end."), False))

    reloaded = SceneStore(store.dir).load_all()
    assert [s.name for s in reloaded] == ["Closing shot", "Opening shot"]  # sorted by name
    assert reloaded[0].prompt_sections["detailed_description"] == "The end."


def test_each_scene_is_its_own_file(store):
    store.save(Scene.from_state("Opening shot", _state(), randomize_seed=True))
    store.save(Scene.from_state("Closing shot", _state(), randomize_seed=True))
    names = sorted(p.name for p in store.dir.glob("*.json"))
    assert names == ["closing-shot.json", "opening-shot.json"]


def test_saved_file_is_readable_json_with_the_name_inside(store):
    scene = store.save(Scene.from_state("Opening shot", _state(), randomize_seed=True))
    data = json.loads(scene.path.read_text(encoding="utf-8"))
    assert data["name"] == "Opening shot"
    assert data["schema_version"] == 1
    assert data["created_at"] and data["updated_at"]


def test_names_that_slugify_alike_get_separate_files(store):
    a = store.save(Scene.from_state("Take 1", _state(), randomize_seed=True))
    b = store.save(Scene.from_state("take-1", _state(), randomize_seed=True))
    assert a.path != b.path
    assert len(list(store.dir.glob("*.json"))) == 2


def test_updating_a_scene_rewrites_the_same_file(store):
    scene = store.save(Scene.from_state("Opening", _state(), randomize_seed=True))
    original_path = scene.path

    scene.capture_from_state(_state(prompt="Rewritten."), randomize_seed=True)
    store.save(scene)

    assert scene.path == original_path
    assert len(list(store.dir.glob("*.json"))) == 1
    assert SceneStore(store.dir).load_all()[0].prompt_sections["detailed_description"] == "Rewritten."


def test_rename_moves_the_file(store):
    scene = store.save(Scene.from_state("Old name", _state(), randomize_seed=True))
    old_path = scene.path

    store.rename(scene, "New name")

    assert not old_path.exists()
    assert scene.path.name == "new-name.json"
    assert SceneStore(store.dir).load_all()[0].name == "New name"


# ---------------------------------------------------------------------------------
# Description
# ---------------------------------------------------------------------------------

def test_description_is_saved_and_reloaded(store):
    store.save(Scene.from_state(
        "Opening", _state(), randomize_seed=True,
        description="Establishing beat before the reveal"))

    reloaded = SceneStore(store.dir).load_all()[0]
    assert reloaded.description == "Establishing beat before the reveal"


def test_description_defaults_to_empty():
    assert Scene.from_state("Bare", _state(), randomize_seed=True).description == ""
    assert Scene.from_dict({"name": "Legacy"}).description == ""


def test_a_scene_file_from_before_descriptions_still_loads(store):
    """Files written by the previous version have no description key at all."""
    store.dir.mkdir(parents=True, exist_ok=True)
    (store.dir / "old.json").write_text(json.dumps({
        "name": "Old scene", "prompt_text": "hi", "duration_seconds": 5.0,
        "seed": 1, "randomize_seed": True, "refs": [],
    }), encoding="utf-8")

    scene = SceneStore(store.dir).load_all()[0]
    assert scene.name == "Old scene" and scene.description == ""


@pytest.mark.parametrize("raw,expected", [
    ("  padded  ", "padded"),
    ("two\nlines\there", "two lines here"),
    ("", ""),
    (None, ""),
    (123, "123"),
])
def test_clean_description(raw, expected):
    assert clean_description(raw) == expected


def test_description_is_capped():
    assert len(clean_description("x" * 500)) == MAX_DESCRIPTION_LENGTH


def test_description_is_not_part_of_the_modified_check():
    """It belongs to the scene, not the editor, so it is saved the moment it is edited."""
    state = _state()
    scene = Scene.from_state("Opening", state, randomize_seed=True, description="first")
    scene.description = "changed"
    assert scene.matches_state(state, randomize_seed=True)


def test_set_details_updates_both_name_and_description(store):
    scene = store.save(Scene.from_state(
        "Old name", _state(), randomize_seed=True, description="old note"))
    old_path = scene.path

    store.set_details(scene, "New name", "new note")

    assert scene.name == "New name"
    assert scene.description == "new note"
    assert not old_path.exists()

    reloaded = SceneStore(store.dir).load_all()[0]
    assert (reloaded.name, reloaded.description) == ("New name", "new note")


def test_set_details_can_change_only_the_description(store):
    scene = store.save(Scene.from_state("Keep me", _state(), randomize_seed=True))
    path_before = scene.path

    store.set_details(scene, "Keep me", "just a note")

    assert scene.path == path_before          # no needless file churn
    assert SceneStore(store.dir).load_all()[0].description == "just a note"


def test_set_details_normalises_what_it_is_given(store):
    scene = store.save(Scene.from_state("Opening", _state(), randomize_seed=True))
    store.set_details(scene, "  Trimmed  ", "  multi\nline  ")
    assert scene.name == "Trimmed"
    assert scene.description == "multi line"


def test_set_details_ignores_an_empty_name(store):
    scene = store.save(Scene.from_state("Opening", _state(), randomize_seed=True))
    store.set_details(scene, "   ", "still fine")
    assert scene.name == "Opening"
    assert scene.description == "still fine"


def test_duplicate_carries_the_description(store):
    original = store.save(Scene.from_state(
        "Opening", _state(), randomize_seed=True, description="the note"))
    assert store.duplicate(original).description == "the note"


def test_duplicate_creates_an_independent_copy(store):
    original = store.save(Scene.from_state("Opening", _state(), randomize_seed=True))
    copy = store.duplicate(original)

    assert copy.name == "Opening copy"
    assert copy.path != original.path
    copy.prompt_sections["summary"] = "diverged"
    store.save(copy)

    reloaded = SceneStore(store.dir)
    reloaded.load_all()
    assert reloaded.find("Opening").prompt_sections["summary"] != "diverged"
    assert reloaded.find("Opening copy").prompt_sections["summary"] == "diverged"


def test_duplicating_twice_keeps_names_unique(store):
    original = store.save(Scene.from_state("Opening", _state(), randomize_seed=True))
    first = store.duplicate(original)
    second = store.duplicate(original)
    assert {first.name, second.name} == {"Opening copy", "Opening copy 2"}


def test_delete_removes_the_file_and_the_entry(store):
    scene = store.save(Scene.from_state("Doomed", _state(), randomize_seed=True))
    path = scene.path
    store.delete(scene)
    assert not path.exists()
    assert store.scenes == []


def test_name_lookup_is_case_insensitive(store):
    store.save(Scene.from_state("Opening", _state(), randomize_seed=True))
    assert store.find("OPENING") is not None
    assert store.name_exists("opening")


def test_name_exists_can_exclude_the_scene_being_renamed(store):
    scene = store.save(Scene.from_state("Opening", _state(), randomize_seed=True))
    assert store.name_exists("Opening", exclude=scene) is False


def test_a_damaged_file_does_not_destroy_the_catalogue(store, caplog):
    store.save(Scene.from_state("Good one", _state(), randomize_seed=True))
    (store.dir / "broken.json").write_text("{not json at all", encoding="utf-8")
    (store.dir / "empty.json").write_text("null", encoding="utf-8")

    reloaded = SceneStore(store.dir).load_all()
    assert [s.name for s in reloaded] == ["Good one"]


def test_missing_directory_loads_as_an_empty_catalogue(tmp_path):
    assert SceneStore(tmp_path / "never-created").load_all() == []


def test_no_temp_files_are_left_behind(store):
    for i in range(5):
        store.save(Scene.from_state(f"Scene {i}", _state(), randomize_seed=True))
    assert not list(store.dir.glob(".scene-*.tmp"))


def test_write_failure_raises_and_leaves_no_temp_file(store, monkeypatch):
    store.dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "harmon3.scenes.json.dump",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))

    with pytest.raises(SceneWriteError):
        store.save(Scene.from_state("Doomed", _state(), randomize_seed=True))
    assert not list(store.dir.glob(".scene-*.tmp"))


# ---------------------------------------------------------------------------------
# Relocating the catalogue
# ---------------------------------------------------------------------------------

def test_resolve_scenes_dir_falls_back_to_the_default():
    assert config.resolve_scenes_dir("") == config.SCENES_DIR
    assert config.resolve_scenes_dir(None) == config.SCENES_DIR
    assert config.resolve_scenes_dir("   ") == config.SCENES_DIR
    assert config.resolve_scenes_dir("C:/elsewhere/scenes") == Path("C:/elsewhere/scenes")


def test_move_to_relocates_every_scene(store, tmp_path):
    source = store.dir
    store.save(Scene.from_state("Alpha", _state(), randomize_seed=True))
    store.save(Scene.from_state("Beta", _state(), randomize_seed=True))
    destination = tmp_path / "elsewhere"

    moved, problems = store.move_to(destination)

    assert (moved, problems) == (2, [])
    assert store.dir == destination
    assert sorted(p.name for p in destination.glob("*.json")) == ["alpha.json", "beta.json"]
    assert list(source.glob("*.json")) == []


def test_moved_scenes_still_load(store, tmp_path):
    store.save(Scene.from_state("Alpha", _state(prompt="kept"), randomize_seed=True))
    destination = tmp_path / "elsewhere"
    store.move_to(destination)

    reloaded = SceneStore(destination).load_all()
    assert [s.name for s in reloaded] == ["Alpha"]
    assert reloaded[0].prompt_sections["detailed_description"] == "kept"


def test_move_to_does_not_overwrite_a_name_already_there(store, tmp_path):
    source = store.dir
    store.save(Scene.from_state("Alpha", _state(prompt="mine"), randomize_seed=True))
    destination = tmp_path / "elsewhere"
    destination.mkdir()
    (destination / "alpha.json").write_text(
        '{"name": "Different Alpha", "prompt_text": "theirs"}', encoding="utf-8")

    moved, problems = store.move_to(destination)

    assert moved == 0
    assert problems and "already exists" in problems[0]
    # Neither file was destroyed: the newcomer stayed put, the incumbent is untouched.
    assert (source / "alpha.json").is_file()
    # The incumbent was written before the split, so its prompt migrates into a section.
    assert SceneStore(destination).load_all()[0].prompt_sections["detailed_description"] == "theirs"


def test_move_to_the_same_folder_is_a_no_op(store):
    store.save(Scene.from_state("Alpha", _state(), randomize_seed=True))
    assert store.move_to(store.dir) == (0, [])
    assert store.count_files() == 1


def test_move_to_from_a_folder_that_never_existed(tmp_path):
    store = SceneStore(tmp_path / "never")
    destination = tmp_path / "target"
    assert store.move_to(destination) == (0, [])
    assert store.dir == destination


def test_count_files(store, tmp_path):
    assert SceneStore(tmp_path / "missing").count_files() == 0
    store.save(Scene.from_state("Alpha", _state(), randomize_seed=True))
    assert store.count_files() == 1


def test_writable_creates_the_folder_and_reports_success(tmp_path):
    target = tmp_path / "brand" / "new"
    store = SceneStore(target)
    assert store.writable() is None
    assert target.is_dir()
    assert not list(target.glob(".harmon3-write-test"))   # the probe cleans up


def test_writable_reports_a_folder_it_cannot_create():
    store = SceneStore(Path("Z:/definitely/not/a/real/drive/scenes"))
    problem = store.writable()
    assert problem and "cannot" in problem


# ---------------------------------------------------------------------------------
# Descriptions and diagnostics
# ---------------------------------------------------------------------------------

def test_ref_summary_counts_by_kind():
    scene = Scene.from_state("Opening", _state(), randomize_seed=True)
    assert scene.ref_summary() == "1 image, 1 video, 1 audio"


def test_ref_summary_with_no_references():
    scene = Scene.from_state("Bare", _state(refs=RefSet()), randomize_seed=True)
    assert scene.ref_summary() == "no references"


def test_prompt_summary_collapses_whitespace_and_truncates():
    scene = Scene.from_state("Long", _state(prompt="a\n\nb   c " + "x" * 200), True)
    summary = scene.prompt_summary(limit=40)
    assert len(summary) == 40 and summary.endswith("...") and "\n" not in summary


def test_prompt_summary_when_empty():
    assert Scene.from_state("Bare", _state(prompt=""), True).prompt_summary() == "(no prompt)"


def test_missing_local_files_is_reported(tmp_path):
    present = tmp_path / "here.png"
    present.write_bytes(b"x")
    state = _state(refs=RefSet(images=[
        RefRow(kind=IMAGE, local_path=str(present)),
        RefRow(kind=IMAGE, local_path=str(tmp_path / "gone.png")),
        RefRow(kind=IMAGE, comfy_name="on_server.png"),
    ]))
    scene = Scene.from_state("Mixed", state, randomize_seed=True)
    assert scene.missing_local_files() == [str(tmp_path / "gone.png")]


# ---------------------------------------------------------------------------------
# Slugs
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("Opening shot", "opening-shot"),
    ("  Trim  Me  ", "trim-me"),
    ("Caf\u00e9 sc\u00e8ne", "cafe-scene"),
    ("!!!", "scene"),
    ("", "scene"),
    ("CON", "scene-con"),          # reserved on Windows
    ("a" * 200, "a" * 60),
])
def test_slugify(name, expected):
    assert slugify(name) == expected


def test_from_dict_rejects_a_nameless_scene():
    with pytest.raises(ValueError):
        Scene.from_dict({"prompt_text": "x"})


def test_from_dict_falls_back_to_the_filename(tmp_path):
    scene = Scene.from_dict({"prompt_text": "x"}, path=tmp_path / "rescued.json")
    assert scene.name == "rescued"


def test_from_dict_tolerates_missing_optional_fields():
    scene = Scene.from_dict({"name": "Sparse"})
    assert scene.duration_seconds == config.DEFAULT_DURATION
    assert scene.refs == []
    assert scene.randomize_seed is True


# ---------------------------------------------------------------------------------
# Projects: the finished video a set of scenes is assembled into
# ---------------------------------------------------------------------------------

def _saved(store, *names):
    return [store.save(Scene(name=name)) for name in names]


def test_a_scene_belongs_to_no_project_until_it_is_filed(store):
    scene, = _saved(store, "Loose")
    assert scene.project == UNGROUPED
    assert store.ungrouped() == [scene]
    assert store.project_names() == []


def test_a_scene_written_before_projects_reads_as_ungrouped():
    """The field is simply absent in an older file; that is not a migration."""
    scene = Scene.from_dict({"name": "Old"})
    assert scene.project == UNGROUPED
    assert scene.project_index == 0


def test_filing_scenes_gives_them_a_running_order(store):
    a, b, c = _saved(store, "A", "B", "C")
    project = store.create_project("Hero film")
    for scene in (a, b, c):
        store.set_project(scene, project)

    assert [s.name for s in store.scenes_in(project)] == ["A", "B", "C"]
    assert [s.project_index for s in store.scenes_in(project)] == [0, 1, 2]


def test_a_scene_can_be_dropped_into_the_middle(store):
    a, b, c = _saved(store, "A", "B", "C")
    project = store.create_project("Hero film")
    store.set_project(a, project)
    store.set_project(b, project)
    store.set_project(c, project, 0)

    assert [s.name for s in store.scenes_in(project)] == ["C", "A", "B"]


def test_the_order_is_renumbered_rather_than_left_with_gaps(store):
    """0, 1, 2 on disk, always -- so a later insert cannot land unpredictably."""
    a, b, c = _saved(store, "A", "B", "C")
    project = store.create_project("Hero film")
    for scene in (a, b, c):
        store.set_project(scene, project)

    store.set_project(b, UNGROUPED)
    assert [s.project_index for s in store.scenes_in(project)] == [0, 1]
    assert [s.name for s in store.scenes_in(project)] == ["A", "C"]


def test_deleting_a_scene_closes_the_gap_it_leaves(store):
    a, b, c = _saved(store, "A", "B", "C")
    project = store.create_project("Hero film")
    for scene in (a, b, c):
        store.set_project(scene, project)

    store.delete(b)
    assert [(s.name, s.project_index) for s in store.scenes_in(project)] \
        == [("A", 0), ("C", 1)]


def test_moving_a_scene_up_and_down_its_project(store):
    a, b, c = _saved(store, "A", "B", "C")
    project = store.create_project("Hero film")
    for scene in (a, b, c):
        store.set_project(scene, project)

    store.move_within_project(c, -1)
    assert [s.name for s in store.scenes_in(project)] == ["A", "C", "B"]
    store.move_within_project(c, -1)
    assert [s.name for s in store.scenes_in(project)] == ["C", "A", "B"]
    store.move_within_project(c, -1)          # already first; stays there
    assert [s.name for s in store.scenes_in(project)] == ["C", "A", "B"]


def test_a_project_and_its_order_survive_a_reload(store):
    a, b = _saved(store, "A", "B")
    project = store.create_project("Hero film")
    store.set_project(b, project)
    store.set_project(a, project)

    reopened = SceneStore(store.dir)
    reopened.load_all()
    assert reopened.project_names() == ["Hero film"]
    assert [s.name for s in reopened.scenes_in("Hero film")] == ["B", "A"]


def test_an_empty_project_survives_a_reload(store):
    """It exists nowhere but the registry, and it is what scenes get dragged onto."""
    store.create_project("Nothing yet")

    reopened = SceneStore(store.dir)
    reopened.load_all()
    assert reopened.project_names() == ["Nothing yet"]


def test_a_project_named_only_by_a_scene_still_appears(store):
    """A scene file arriving from elsewhere brings its project with it."""
    scene, = _saved(store, "Imported")
    scene.project = "Another film"
    store._write(scene)

    reopened = SceneStore(store.dir)
    reopened.load_all()
    assert reopened.project_names() == ["Another film"]


def test_the_registry_is_not_mistaken_for_a_scene(store):
    _saved(store, "A")
    store.create_project("Hero film")

    reopened = SceneStore(store.dir)
    reopened.load_all()
    assert [s.name for s in reopened.scenes] == ["A"]
    assert reopened.count_files() == 1


def test_an_unreadable_registry_costs_only_the_empty_projects(store):
    a, = _saved(store, "A")
    project = store.create_project("Hero film")
    store.set_project(a, project)
    store.create_project("Empty one")
    (store.dir / PROJECTS_FILENAME).write_text("{ not json", encoding="utf-8")

    reopened = SceneStore(store.dir)
    reopened.load_all()
    assert reopened.project_names() == ["Hero film"]      # rebuilt from the scene
    assert [s.name for s in reopened.scenes_in("Hero film")] == ["A"]


def test_renaming_a_project_carries_its_scenes(store):
    a, b = _saved(store, "A", "B")
    project = store.create_project("Hero film")
    for scene in (a, b):
        store.set_project(scene, project)

    store.rename_project(project, "Hero film final")
    assert store.project_names() == ["Hero film final"]
    assert [s.name for s in store.scenes_in("Hero film final")] == ["A", "B"]

    reopened = SceneStore(store.dir)
    reopened.load_all()
    assert [s.project for s in reopened.scenes] == ["Hero film final"] * 2


def test_deleting_a_project_keeps_every_scene(store):
    """A project arranges work; removing the arrangement must not remove the work."""
    a, b = _saved(store, "A", "B")
    project = store.create_project("Hero film")
    for scene in (a, b):
        store.set_project(scene, project)

    released = store.delete_project(project)
    assert released == 2
    assert store.project_names() == []
    assert {s.name for s in store.ungrouped()} == {"A", "B"}
    assert store.count_files() == 2


def test_two_projects_cannot_share_a_name(store):
    first = store.create_project("Hero film")
    second = store.create_project("Hero film")
    assert first == "Hero film"
    assert second == "Hero film 2"


def test_renaming_onto_a_taken_name_is_made_unique(store):
    store.create_project("Hero film")
    other = store.create_project("Second film")
    used = store.rename_project(other, "Hero film")
    assert used == "Hero film 2"


def test_a_blank_project_name_is_not_a_project(store):
    assert store.create_project("   ") == UNGROUPED
    assert store.project_names() == []


def test_a_duplicate_lands_beside_its_original(store):
    """A duplicate is the next take of that shot, not the last shot of the video."""
    a, b = _saved(store, "A", "B")
    project = store.create_project("Hero film")
    store.set_project(a, project)
    store.set_project(b, project)

    copy = store.duplicate(a)
    assert [s.name for s in store.scenes_in(project)] == ["A", "A copy", "B"]
    assert copy.project == project


def test_a_project_is_not_part_of_a_scenes_signature(store):
    """Reordering a project is not an edit to the shots in it."""
    a, b = _saved(store, "A", "B")
    project = store.create_project("Hero film")
    before = a.signature()
    store.set_project(a, project)
    store.set_project(b, project)
    store.move_within_project(a, 1)
    assert a.signature() == before

"""
Tests for bundle_frontend.py.

The bundle is the deliverable: one HTML file that opens from disk with no server, no
build step and no network on load. So what is tested is that it really is self-
contained, that the data survives being embedded, and that a template which does not
honour the contract fails loudly instead of shipping a page with no data in it.
"""

import json
import os
import re
import zipfile

import pytest

import bundle_frontend as bundler


@pytest.fixture
def result():
    """A groups build, which is what a first build of any competition produces."""
    from test_build_competition import make_result
    return make_result(n_teams=3, n_golfers=20)


@pytest.fixture
def live_result():
    """The same competition once ESPN has published a field, so the page has scoring."""
    from test_build_competition import golfer_name, live_stage, make_result
    return make_result(n_teams=3, n_golfers=20,
                       espn=live_stage([golfer_name(i) for i in range(20)]))


@pytest.fixture
def template(tmp_path):
    """A minimal template that honours the contract."""
    d = tmp_path / "tpl"
    d.mkdir()
    (d / "style.css").write_text("body { color: rebeccapurple; }")
    (d / "app.js").write_text("var DATA = JSON.parse("
                              "document.getElementById('competition-data').textContent);")
    (d / "logo.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    (d / "index.html").write_text(
        '<!doctype html><html><head><title>{{league_name}} - {{tournament}}</title>'
        '<link rel="stylesheet" href="style.css"></head><body>'
        '<img src="logo.svg" alt=""><p>{{market}} / {{team_count}} / {{competition_id}}</p>'
        f'<script id="competition-data" type="application/json">{bundler.JSON_MARKER}</script>'
        '<script src="app.js"></script></body></html>')
    return str(d)


def read_html(paths):
    with open(paths[0], encoding="utf-8") as f:
        return f.read()


def embedded_json(markup):
    m = re.search(r'<script id="competition-data" type="application/json">(.*?)</script>',
                  markup, re.S)
    assert m, "the data script tag is gone"
    return json.loads(m.group(1).replace("<\\/", "</").replace("<\\!--", "<!--"))


# ---------------------------------------------------------------------------
# Self-containment
# ---------------------------------------------------------------------------

def test_nothing_local_is_left_to_fetch(result, template, tmp_path):
    markup = read_html(bundler.bundle(result, template, str(tmp_path / "out"))[0])
    assert not re.search(r'<link[^>]*rel=["\']stylesheet', markup)
    assert not re.search(r'<script[^>]+src=', markup)
    assert not re.search(r'<img[^>]+src=["\'](?!data:)', markup)
    assert "rebeccapurple" in markup
    assert "getElementById('competition-data')" in markup
    assert "data:image/svg+xml;base64," in markup


def test_remote_references_are_left_alone(result, template, tmp_path):
    index = os.path.join(template, "index.html")
    with open(index) as f:
        markup = f.read()
    with open(index, "w") as f:
        f.write(markup.replace("<body>", '<body><img src="https://example.com/x.png" alt="">'))
    out = read_html(bundler.bundle(result, template, str(tmp_path / "out"))[0])
    assert 'src="https://example.com/x.png"' in out


def test_a_missing_asset_is_reported_and_left_as_is(result, template, tmp_path, capsys):
    os.remove(os.path.join(template, "style.css"))
    paths, report = bundler.bundle(result, template, str(tmp_path / "out"))
    assert report["missing"] == ["style.css"]
    assert "style.css" in capsys.readouterr().out
    assert 'href="style.css"' in read_html(paths)     # honest breakage beats a silent drop


# ---------------------------------------------------------------------------
# The data survives embedding
# ---------------------------------------------------------------------------

def test_the_whole_result_is_embedded_and_parses(result, template, tmp_path):
    data = embedded_json(read_html(bundler.bundle(result, template, str(tmp_path / "out"))[0]))
    assert data == result


def test_a_script_tag_in_the_data_cannot_end_the_element(result, template, tmp_path):
    """
    The failure this escaping exists for: one team called "</script>" truncates the
    page's entire dataset, and the page renders empty with no error anywhere.
    """
    result["teams"][0]["team_name"] = "</script><script>alert(1)</script>"
    result["teams"][1]["team_name"] = "<!-- comment --> FC"
    markup = read_html(bundler.bundle(result, template, str(tmp_path / "out"))[0])
    assert "</script><script>alert(1)" not in markup
    data = embedded_json(markup)
    assert data["teams"][0]["team_name"] == "</script><script>alert(1)</script>"
    assert data["teams"][1]["team_name"] == "<!-- comment --> FC"


def test_non_ascii_survives(result, template, tmp_path):
    result["golfers"][0]["name"] = "Rasmus Højgaard"
    data = embedded_json(read_html(bundler.bundle(result, template, str(tmp_path / "out"))[0]))
    assert data["golfers"][0]["name"] == "Rasmus Højgaard"


def test_json_for_script_neutralises_both_hazards():
    out = bundler.json_for_script({"a": "</script>", "b": "<!--"})
    assert "</script>" not in out and "<!--" not in out
    assert json.loads(out.replace("<\\/", "</").replace("<\\!--", "<!--")) == {
        "a": "</script>", "b": "<!--"}


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

def test_tokens_are_substituted(result, template, tmp_path):
    markup = read_html(bundler.bundle(result, template, str(tmp_path / "out"))[0])
    assert "<title>Test - Wyndham Championship</title>" in markup
    assert "Outright Winner / 3 /" in markup
    assert "{{" not in markup


def test_tokens_are_html_escaped(result, template, tmp_path):
    """They come from a hand-written league file and land in markup, not a text node."""
    result["league"]["league_name"] = 'Bogey "Boys" & <b>Co</b>'
    markup = read_html(bundler.bundle(result, template, str(tmp_path / "out"))[0])
    assert "<b>Co</b>" not in markup
    assert "&lt;b&gt;Co&lt;/b&gt;" in markup


def test_an_unknown_token_is_left_alone(result, template, tmp_path):
    index = os.path.join(template, "index.html")
    with open(index) as f:
        markup = f.read()
    with open(index, "w") as f:
        f.write(markup.replace("<body>", "<body>{{not_a_thing}}"))
    assert "{{not_a_thing}}" in read_html(bundler.bundle(result, template, str(tmp_path / "out"))[0])


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------

def test_a_template_without_the_marker_is_refused(result, tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    (d / "index.html").write_text("<!doctype html><html><body>no marker</body></html>")
    with pytest.raises(SystemExit, match="data marker"):
        bundler.bundle(result, str(d), str(tmp_path / "out"))


def test_a_template_without_an_index_is_refused(result, tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(SystemExit, match="index.html"):
        bundler.bundle(result, str(d), str(tmp_path / "out"))


# ---------------------------------------------------------------------------
# The zip
# ---------------------------------------------------------------------------

def test_the_zip_carries_the_page_the_data_and_a_manifest(live_result, template, tmp_path):
    paths, _ = bundler.bundle(live_result, template, str(tmp_path / "out"))
    with zipfile.ZipFile(paths[1]) as z:
        assert sorted(z.namelist()) == ["MANIFEST.txt", "index.html", "result.json"]
        assert json.loads(z.read("result.json")) == live_result
        manifest = z.read("MANIFEST.txt").decode()
    assert live_result["competition_id"] in manifest
    assert live_result["sources"]["kalshi"]["markets_endpoint"] in manifest
    assert live_result["live"]["espn_leaderboard_url"] in manifest


def test_the_manifest_does_not_promise_scoring_a_groups_page_cannot_do(result, template,
                                                                      tmp_path):
    """
    The manifest travels inside the zip and is the only thing in there that says what
    the page does. A groups page fetches nothing at all, so a manifest naming a
    leaderboard endpoint and announcing live scores is a lie that ships -- and one
    nothing would catch, because prose does not raise.

    The failure it causes is somebody waiting all afternoon for a number to appear.
    """
    paths, _ = bundler.bundle(result, template, str(tmp_path / "out"))
    with zipfile.ZipFile(paths[1]) as z:
        manifest = z.read("MANIFEST.txt").decode()
    assert "build mode      groups" in manifest
    assert "It fetches NOTHING" in manifest
    assert "espn.com" not in manifest
    assert "live scoring    none" in manifest


def test_a_groups_result_bundles_without_a_half_written_directory(result, template, tmp_path):
    """
    The manifest is written after the HTML, so anything in it that assumed a live build
    used to fail with the page already on disk -- a dist/ directory holding a scoreboard
    and no zip, plus a traceback pointing at the bundler rather than at the result.
    """
    out = tmp_path / "out"
    paths, report = bundler.bundle(result, template, str(out))
    assert report["missing"] == []
    assert sorted(os.path.basename(p) for p in paths) == [
        "test-wyndham-championship.html", "test-wyndham-championship.zip"]
    assert all(os.path.getsize(p) for p in paths)


def test_the_result_can_be_left_out_of_the_zip(result, template, tmp_path):
    paths, _ = bundler.bundle(result, template, str(tmp_path / "out"), keep_result=False)
    with zipfile.ZipFile(paths[1]) as z:
        assert "result.json" not in z.namelist()


def test_the_basename_says_which_league_and_which_tournament(result, template, tmp_path):
    paths, _ = bundler.bundle(result, template, str(tmp_path / "out"))
    assert os.path.basename(paths[0]) == "test-wyndham-championship.html"
    assert os.path.basename(paths[1]) == "test-wyndham-championship.zip"


def test_an_explicit_basename_wins(result, template, tmp_path):
    paths, _ = bundler.bundle(result, template, str(tmp_path / "out"), basename="week-12")
    assert os.path.basename(paths[0]) == "week-12.html"


# ---------------------------------------------------------------------------
# The real template
# ---------------------------------------------------------------------------

def test_the_shipped_template_honours_its_own_contract(result, tmp_path):
    paths, report = bundler.bundle(result, bundler.DEFAULT_TEMPLATE, str(tmp_path / "out"))
    markup = read_html(paths)
    assert report["missing"] == []
    assert sorted(report["inlined"]) == ["app.js", "lib.js", "style.css"]
    assert embedded_json(markup) == result
    assert not re.search(r'<script[^>]+src=', markup)
    assert "GolfPool" in markup and "computeStandings" in markup

"""
Tests for bundle_frontend.py.

The bundle is the deliverable: one HTML file that opens from disk with no server, no
build step and no network on load. So what is tested is that it really is self-
contained, that the data survives being embedded, and that a template which does not
honour the contract fails loudly instead of shipping a page with no data in it.
"""

import base64
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


@pytest.fixture
def art_template(tmp_path):
    """The same, plus the optional half of the contract: an element for the league's
    art. Kept apart from `template` so the tests above go on proving that a template
    which draws no masthead still bundles."""
    d = tmp_path / "art-tpl"
    d.mkdir()
    (d / "index.html").write_text(
        '<!doctype html><html><body><img id="league-logo" hidden>'
        f'<script id="competition-data" type="application/json">{bundler.JSON_MARKER}</script>'
        f'<script id="league-art" type="application/json">{bundler.ART_MARKER}</script>'
        "</body></html>")
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
# The league's art
#
# The one thing the page carries that the result JSON does not. The result names the
# art with a slug; this is the step that turns that name into two data: URIs, and it
# is the only step in the pipeline that opens the files at all.
# ---------------------------------------------------------------------------

PIXEL = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAA"
                         "DUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


@pytest.fixture
def leagues(tmp_path):
    """A leagues directory with one slug in it, carrying both images."""
    art = tmp_path / "leagues" / "wcw"
    art.mkdir(parents=True)
    (art / "logo.png").write_bytes(PIXEL)
    (art / "banner.png").write_bytes(PIXEL)
    return str(tmp_path / "leagues")


def embedded_art(markup):
    m = re.search(r'<script id="league-art" type="application/json">(.*?)</script>',
                  markup, re.S)
    assert m, "the art script tag is gone"
    return json.loads(m.group(1))


def test_the_slug_becomes_two_data_uris_in_the_page(result, art_template, leagues, tmp_path):
    result["league"]["logo"] = "wcw"
    markup = read_html(bundler.bundle(result, art_template, str(tmp_path / "out"),
                                      leagues_dir=leagues)[0])
    art = embedded_art(markup)
    assert art["logo"].startswith("data:image/png;base64,")
    assert art["banner"].startswith("data:image/png;base64,")


def test_the_images_go_into_the_page_and_not_into_the_data(result, art_template, leagues,
                                                           tmp_path):
    """
    The whole point. The page has to be portable, so it gets the bytes; the result JSON
    beside it in the zip is the input to a rebuild and stays a document about a
    competition, carrying the name of the art and none of it.
    """
    result["league"]["logo"] = "wcw"
    paths, _ = bundler.bundle(result, art_template, str(tmp_path / "out"), leagues_dir=leagues)
    assert embedded_json(read_html(paths))["league"]["logo"] == "wcw"
    with zipfile.ZipFile(paths[1]) as z:
        written = json.loads(z.read("result.json"))
    assert written == result
    assert "data:image" not in json.dumps(written["league"])


def test_a_league_with_no_art_gets_an_empty_object(result, art_template, tmp_path):
    """`{}` and not the marker. A JavaScript comment where a page's JSON should be
    takes the masthead and the whole script with it."""
    markup = read_html(bundler.bundle(result, art_template, str(tmp_path / "out"))[0])
    assert embedded_art(markup) == {}
    assert bundler.ART_MARKER not in markup


def test_half_the_art_is_half_the_keys(result, art_template, leagues, tmp_path):
    os.remove(os.path.join(leagues, "wcw", "banner.png"))
    result["league"]["logo"] = "wcw"
    markup = read_html(bundler.bundle(result, art_template, str(tmp_path / "out"),
                                      leagues_dir=leagues)[0])
    assert set(embedded_art(markup)) == {"logo"}


def test_a_slug_that_resolves_to_nothing_is_reported(result, art_template, leagues,
                                                     tmp_path, capsys):
    """The build recorded art this export cannot find, which means a masthead somebody
    is expecting and will not get."""
    result["league"]["logo"] = "ghost"
    bundler.bundle(result, art_template, str(tmp_path / "out"), leagues_dir=leagues)
    assert "holds no logo or banner" in capsys.readouterr().out


def test_a_template_that_draws_no_art_bundles_untouched(result, template, leagues, tmp_path):
    """The art element is the one optional half of the contract: a template that has no
    masthead never has to mention it, and asking for art it cannot draw is not an error."""
    result["league"]["logo"] = "wcw"
    markup = read_html(bundler.bundle(result, template, str(tmp_path / "out"),
                                      leagues_dir=leagues)[0])
    assert "league-art" not in markup
    assert embedded_json(markup)["league"]["logo"] == "wcw"


def test_the_manifest_names_the_art_it_inlined(result, art_template, leagues, tmp_path):
    """It is the one difference between the page and the result.json beside it, and
    most of a branded page's weight."""
    result["league"]["logo"] = "wcw"
    paths, _ = bundler.bundle(result, art_template, str(tmp_path / "out"), leagues_dir=leagues)
    with zipfile.ZipFile(paths[1]) as z:
        manifest = z.read("MANIFEST.txt").decode()
    assert "league art      logo" in manifest and "banner" in manifest


def test_heavy_art_is_inlined_anyway_and_said_out_loud(result, art_template, leagues,
                                                       tmp_path, capsys):
    """Somebody who wants a 3 MB banner in their scoreboard is entitled to one -- a page
    too heavy to email is a nuisance, not a wrong answer. They should hear about it from
    the build rather than from an inbox."""
    with open(os.path.join(leagues, "wcw", "banner.png"), "wb") as f:
        f.write(b"\x89PNG" + b"\x00" * (bundler.HEAVY_ART_BYTES + 1))
    result["league"]["logo"] = "wcw"
    markup = read_html(bundler.bundle(result, art_template, str(tmp_path / "out"),
                                      leagues_dir=leagues)[0])
    assert embedded_art(markup)["banner"].startswith("data:image/png;base64,")
    assert "lands in the page" in capsys.readouterr().out


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
# The templates that actually ship
#
# Two of them, and the contract is the contract for both: the designed page
# (frontend/scoreboard, the default) and the plain reference (frontend/template).
# ---------------------------------------------------------------------------

SHIPPED = [bundler.DEFAULT_TEMPLATE, bundler.REFERENCE_TEMPLATE]
SHIPPED_IDS = ["scoreboard", "reference"]


def test_the_default_is_the_designed_page():
    """
    `python bundle_frontend.py --result ...` with no --template is the whole of what
    the skill runs, so what it produces has to be the page somebody wants to be handed
    rather than the one that exists to prove the contract.
    """
    assert os.path.basename(bundler.DEFAULT_TEMPLATE) == "scoreboard"
    assert os.path.basename(bundler.REFERENCE_TEMPLATE) == "template"


@pytest.mark.parametrize("shipped", SHIPPED, ids=SHIPPED_IDS)
def test_a_shipped_template_honours_its_own_contract(result, shipped, tmp_path):
    paths, report = bundler.bundle(result, shipped, str(tmp_path / "out"))
    markup = read_html(paths)
    assert report["missing"] == []
    assert sorted(report["inlined"]) == ["../lib.js", "app.js", "style.css"]
    assert embedded_json(markup) == result
    assert not re.search(r'<script[^>]+src=', markup)
    assert not re.search(r'<link[^>]*rel=["\']stylesheet', markup)
    assert "GolfPool" in markup and "computeStandings" in markup

    # Every token the bundler knows how to fill has been filled. Not a blanket "{{" ban:
    # an unrecognised token is deliberately left alone, and both templates carry a
    # comment about `{{tokens}}` that is documentation rather than a placeholder.
    assert not re.search(r"\{\{\s*(?:%s)\s*\}\}" % "|".join(
        ["league_name", "tournament", "market", "generated_at", "team_count",
         "competition_id"]), markup)
    assert f"<title>{result['league']['league_name']}" in markup
    assert result["competition_id"] in markup

    # Both shipped pages draw a masthead, so both carry the art element -- and it comes
    # out as JSON rather than as the marker. A page left holding a JavaScript comment
    # where its art should be dies on the first line of its own script.
    assert embedded_art(markup) == {}          # the fixture league has no art
    assert bundler.ART_MARKER not in markup


@pytest.mark.parametrize("shipped", SHIPPED, ids=SHIPPED_IDS)
def test_a_shipped_template_reaches_no_host_at_all_from_its_markup(result, shipped, tmp_path):
    """
    Every remote reference in a template survives bundling untouched and is then
    requested on every open -- and fails on the first one that happens offline, which
    for this page is most of them. The design was drawn against Google Fonts; the
    stylesheet names the families and falls back to a stack instead.
    """
    markup = read_html(bundler.bundle(result, shipped, str(tmp_path / "out"))[0])
    assert not re.search(r'<(?:link|script|img)[^>]+(?:href|src)=["\']//', markup)
    assert not re.search(r'<(?:link|script|img)[^>]+(?:href|src)=["\']https?://', markup)


@pytest.mark.parametrize("shipped", SHIPPED, ids=SHIPPED_IDS)
def test_no_shipped_template_still_knows_how_to_draw_a_price_moving(result, shipped, tmp_path):
    """
    A markup-level guard, because the two render suites can only prove a page does not
    draw movement for the data they happen to feed it. This proves the code to draw it
    is not in the bundle at all -- on both templates, in one place, whatever a future
    result file happens to contain.
    """
    markup = read_html(bundler.bundle(result, shipped, str(tmp_path / "out"))[0])
    for gone in ("MOVEMENT", "moveCell", "g-move", "gmove",
                 "odds_snapshot.refreshed", "odds.current", "refresh-odds"):
        assert gone not in markup, gone


@pytest.mark.parametrize("shipped", SHIPPED, ids=SHIPPED_IDS)
def test_a_shipped_template_carries_the_one_copy_of_the_standings_rule(shipped):
    """
    `../lib.js`, above both template directories. Two copies of the rule that decides
    the pool is one implementation and one rumour, and only one of them would be under
    the parity test.
    """
    with open(os.path.join(shipped, "index.html"), encoding="utf-8") as f:
        assert '<script src="../lib.js"></script>' in f.read()
    assert not os.path.exists(os.path.join(shipped, "lib.js"))


def test_an_image_whose_source_arrives_at_runtime_does_not_break_the_bundle(result, tmp_path):
    """
    `<img src="">` is what an element fed from the data looks like before anybody
    thinks about it. It joins to the template directory, which exists -- so an
    exists() check sends the bundler off to base64-encode a directory and it dies
    with IsADirectoryError three frames down.
    """
    d = tmp_path / "tpl"
    d.mkdir()
    (d / "index.html").write_text(
        '<!doctype html><html><body><img src="" alt=""><img alt="">'
        f'<script id="competition-data" type="application/json">{bundler.JSON_MARKER}</script>'
        "</body></html>")
    paths, report = bundler.bundle(result, str(d), str(tmp_path / "out"))
    assert report["missing"] == []
    assert 'src=""' in read_html(paths)

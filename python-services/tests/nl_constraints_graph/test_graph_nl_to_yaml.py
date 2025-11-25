import os
import sys

# --- Ensure python-services is on sys.path ---
CURRENT_DIR = os.path.dirname(__file__)                      # .../python-services/tests/nl_constraints_graph
PYTHON_SERVICES_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PYTHON_SERVICES_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_SERVICES_ROOT)

import nl_constraints_graph.graph_nl_to_yaml as mod


def test_build_graph_returns_app_with_graph():
    """
    Basic sanity: build_graph should return an object with get_graph(),
    and the graph should at least support draw_mermaid() or draw().
    """
    app = mod.build_graph()
    # Should have get_graph method like in export_graph_png
    assert hasattr(app, "get_graph")

    g = app.get_graph()
    # Graph should have at least one of these methods
    assert hasattr(g, "draw") or hasattr(g, "draw_mermaid")


def test_export_graph_png_mermaid_fallback(tmp_path, monkeypatch):
    """
    When draw() fails or returns None, export_graph_png should fall back to
    draw_mermaid() and write a .mermaid file.
    """

    class FakeGraph:
        def draw(self):
            # Simulate GraphViz not available / error
            raise Exception("graphviz not installed")

        def draw_mermaid(self):
            return "graph TD;\n  intent --> validate;\n"

    class FakeApp:
        def get_graph(self):
            return FakeGraph()

    # Make build_graph() return our fake app
    monkeypatch.setattr(mod, "build_graph", lambda: FakeApp())

    out_path = tmp_path / "test_graph.png"  # even if .png, should fall back to .mermaid
    result = mod.export_graph_png(str(out_path))

    # Should end with .mermaid after fallback
    assert result.endswith(".mermaid")
    assert os.path.exists(result)

    with open(result, "r", encoding="utf-8") as f:
        content = f.read()

    assert "intent" in content
    assert "validate" in content


def test_export_graph_png_uses_graphviz_when_available(tmp_path, monkeypatch):
    """
    When draw() returns an object with render(), export_graph_png should call it
    and produce a .png file.
    """

    rendered_paths = {}

    class FakeDot:
        def render(self, base, format="png", cleanup=True):
            # Simulate GraphViz writing a file
            png_path = base + ".png"
            with open(png_path, "wb") as f:
                f.write(b"fake png data")
            rendered_paths["path"] = png_path
            return png_path

    class FakeGraph:
        def draw(self):
            # Return something that looks like a graphviz.Digraph
            return FakeDot()

        def draw_mermaid(self):
            raise AssertionError("draw_mermaid should not be called when draw() works")

    class FakeApp:
        def get_graph(self):
            return FakeGraph()

    monkeypatch.setattr(mod, "build_graph", lambda: FakeApp())

    out_path = tmp_path / "graph_output.png"
    result = mod.export_graph_png(str(out_path))

    # Should keep .png extension
    assert result.endswith(".png")
    assert os.path.exists(result)
    # And our fake render should have run
    assert rendered_paths["path"] == result
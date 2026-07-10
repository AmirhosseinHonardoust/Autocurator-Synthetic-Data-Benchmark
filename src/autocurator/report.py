from importlib import resources
from pathlib import Path

from jinja2 import Template


def _default_template_text() -> str:
    """Load the bundled report template (works from a wheel or editable install)."""
    return (
        resources.files("autocurator").joinpath("templates/report.html").read_text(encoding="utf-8")
    )


def render_report(out_path: str, context: dict, template_path: str | None = None) -> None:
    if template_path:
        tpl_text = Path(template_path).read_text(encoding="utf-8")
    else:
        tpl_text = _default_template_text()
    html = Template(tpl_text).render(**context)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
